# ace-jax in LAMMPS: why throughput falls above 16k atoms

Diagnosis on a Modal A100-80GB, float64, Cantor medium, at b7b7605
(`perf/ace-jax-speedups`). The lammps-jax build in the image is 4a7f4fb.
Scripts are in `bench/perf/` (`bundle_scaling.py`, `lammps_bundle_variant.py`,
`modal_bundle_scaling.py`). Results are in `bench/perf/results/bundle_scaling_*.json`.

## Symptom

In `bench/scaling/results/modal-a100.jsonl` (LAMMPS, `pair_style jax/kk`),
ace-jax throughput peaks at 16k atoms and then falls:

| atoms | PACE, LAMMPS | PACE, standalone | ACE, LAMMPS | ACE, standalone |
|---|---|---|---|---|
| 16k | 14.5 ms | 8.1 ms | 35 ms | 21 ms |
| 65k | 64 ms | 34 ms | 373 ms | 96 ms |
| 131k | 147 ms | 67 ms | 2269 ms (sparse) | 189 ms |
| 262k | 402 ms | 135 ms | OOM | 373 ms |

Times are per step. ML-PACE in LAMMPS stays flat at about 2.3M atom-steps/s.
The standalone column is the calculator's `force_s`.

## Measurements

**LAMMPS overhead is negligible.** I timed the bundle's per-step program
outside LAMMPS: `value_and_grad` of the masked energy over all `max_atoms`
positions, as `lammps_jax.export.wrap_energy_fn` builds it. Its inputs are
lammps-jax-shaped, built from the real supercell and sized by
`run_lammps.capacity`. For PACE it matches the LAMMPS step time to within 10%
at every size: 5.2, 13.9, 62.7, 146 and 398 ms at 4k, 16k, 65k, 131k and 262k
atoms, against 4.7, 14.5, 64, 147 and 402 ms in LAMMPS. The neighbour list,
packing kernels, host work and copies are not the problem.

**Breakdown of the stock PACE bundle** (ms per call):

| atoms | full | packing (argsort + scatters) | model fwd+bwd, unchunked | model fwd+bwd, 32k-row blocks |
|---|---|---|---|---|
| 16k | 13.9 | 1.9 | 8.7 | 8.2 |
| 65k | 62.7 | 5.9 | 36.6 | 43.3 |
| 131k | 146 | 11.6 | 93.6 | 85.2 |
| 262k | 398 | 23.3 | 291 | 176 |

- The packing costs about 0.09 µs per atom at every size. It is linear, and
  about 6–13% of a step.
- The rest (the ghost-position gradient and the scatter into the dense slots)
  is also roughly linear: 0.2–0.3 µs per atom.
- Only the model's own forward and backward pass grows faster than N. It costs
  0.53 µs per atom at 16k and 1.11 µs per atom at 262k.

**The superlinear term is the backward pass of the product-basis gather.**
Per-kernel traces (`bundle_scaling_trace*.json`) show where the time goes:

- **PACE.** The gather in `_node_energies_t` is `At[s.T]`, on the
  feature-major `(NZ·n_a, n)` A. XLA lowers its adjoint to three
  `input_scatter_fusion` kernels on `(n_rows, 180)`. They take 9.4 of 35.6 ms
  at 65k and 190 of 291 ms at 262k. That is 0.13 µs per row at 65k and
  0.66 µs per row at 262k. All other kernels together stay at
  0.35 µs per row at both sizes.
- **ACE.** The same kind of kernel, on `(157, n_rows)`, takes 88 of 207 ms
  at 65k unchunked. The same work in 32k-row blocks takes 16.7 ms.

I have not established why XLA's scatter slows per row as the buffer grows.

**Why only LAMMPS sees it.** The calculator evaluates the dense model in
blocks of `CHUNK_NODES` rows: `energy_forces_virial_dense`, using `lax.map`
plus `jax.checkpoint`. The bundle calls `site_energies_dense` on all
`max_owned` rows in one piece. `make_energy_fn` has no chunking at all, so the
chunking hypothesis in the brief (candidate 3) has the direction reversed.
The unchunked bundle also needs memory that grows with N:

- PACE needs 1.4, 5.9, 11.8 and 24 GB of XLA temp at 16k, 65k, 131k and
  262k. That is why 524k runs out of memory.
- ACE needs 19.5 GB at 65k. At 131k the dense ACE program runs out of memory
  even with the whole card free. The sweep then falls back to the sparse
  layout, which gives the 2.27 s step.

**Slot sizing costs a constant factor, but rcut-sized slots are not safe
on the benchmark deck.** The bundle sizes `k_dense` and `max_edges` for
rcut + skin: 86 slots, where Cantor's coordination within rcut = 5.0 Å is 42
on the lattice. The bundle's `contract.cutoff` is 5.0 for the ACE and PACE
models alike, equal to the rcut `capacity` uses, so no pair basis reaches
further. lammps-jax 4a7f4fb packs only pairs within the cutoff
(`PackNeighborFunctor` rejects `d² > cutsq`). A bundle with k(rcut) + 8 = 50
slots therefore runs, and it is 1.5× faster. It ran on short runs (60 steps)
at every size up to 524k atoms.

The Task 11 overflows were still real. On the benchmark deck (50 warm-up plus
200 timed steps, `bench/perf/results/bundle_scaling_lammps_task11*.json`):

- `acejax-ace/Cantor/small` and `acejax-pace/Cantor/medium` at 256 atoms
  finished with 50 slots.
- `acejax-ace/Cantor/large` at 256 atoms aborted at step 175–180 with
  "global max 19968 edges, capacity 14100", the Task 11 message.

The cause is the models, not skin pairs:

- **The benchmark structures collapse.** The deck starts from a perfect
  lattice with no velocities. Random species give nonzero forces, and these
  models drive the atoms together. For ACE Cantor large, the potential energy
  falls from −355 to −609 eV over 250 steps. The largest coordination within
  rcut, from `compute coord/atom`, rises from 42 to 50. For ACE Cantor small
  it rises from 42 to 48 by step 250, and is still rising. There were no
  neighbour-list rebuilds.
- **Once one atom passes k_dense, the step fails as designed.** The energy
  function returns NaN energies and forces, so positions go NaN.
- **NaN distances then pass lammps-jax's cutoff test.** `d² > cutsq` is false
  for NaN, so the next pack keeps every listed pair: 19968 = 256 × 78, which
  is exactly LAMMPS's "Total # of neighbors", the rcut + skin list.

The N × 78 count in every Task 11 failure is therefore the list size seen
through NaN positions. It is not evidence that skin pairs are packed. I
inferred the NaN step from the code: thermo was every 5 steps, and the NaN
itself never reached a thermo line. I did not re-run PACE Cantor large at 16k
atoms. Its energy fell the same way (718 to 282 eV), so it is probably the
same mechanism.

**Ruled out as the decline's cause:**

- The argsort (candidate 1): linear, and at most 13% of a step.
- The ghost-row gradient and scatter-based force assembly (candidate 2):
  linear.
- LAMMPS and lammps-jax overhead (candidate 5).

**Partly a cause: memory (candidate 4).** It is a consequence of the missing
chunking. It is also a plausible explanation for the one unexplained gap:
ACE at 65k takes 373 ms in LAMMPS, against 239 ms for the same program alone.
The chunked bundle removes that gap too.

## Fix and measured gain

End to end in LAMMPS, 60 steps, identical first-step energies
(`bundle_scaling_lammps_*.json`). Numbers are atom-steps/s.

| model, atoms | stock | chunked (safe) | chunked + rcut slots (not safe on this deck) |
|---|---|---|---|
| PACE 16k | 1.13M | 1.13M | **1.77M** (1.56×) |
| PACE 131k | 0.88M | 1.12M (1.27×) | **1.81M** (2.06×) |
| PACE 262k | 0.66M | 1.10M (1.68×) | **1.81M** (2.74×) |
| PACE 524k | OOM | — | **1.77M** |
| ACE 16k | 450k | 468k | **658k** (1.46×) |
| ACE 65k | 175k | 482k (2.75×) | **738k** (4.2×) |
| ACE 131k | 58k (sparse; dense OOM) | 448k (7.7×) | **686k** (11.9×) |

- The PACE chunk is 65,536 rows. The ACE chunk is 32,768 rows.
- With both changes, throughput is flat from 16k to 524k atoms. PACE at 262k
  (145 ms per step) is within 7% of the standalone calculator (135 ms).
- ML-PACE is still 1.3× faster.

**Recommendation:**

1. **Chunk the bundle.** `make_energy_fn` (dense) should evaluate
   `site_energies_dense` in `lax.map` blocks under `jax.checkpoint`, as the
   calculator does. Prototype: `bundle_scaling.make_chunked_energy_fn`.
   - Use about 32k rows per block, not `CHUNK_NODES` = 16384. With
     `max_owned` = 1.1 N, a 16k-atom system then stays one block.
     `CHUNK_NODES` would split it into two blocks and pay the checkpoint
     recompute. That costs 1.5× for both models: 20.9 ms against 13.9 ms for
     PACE.
   - Measured on the bundle program alone, 32k and 64k blocks are equal for
     PACE: 275 against 270 ms at 262k. 32k is better for ACE: 176 against
     206 ms at 65k.
   - Keep the checkpoint. Without it the program is slower than unchunked and
     uses more memory: 98 against 63 ms and 14.6 GB at 65k for PACE, and
     ACE runs out of memory.
   - Chunking also lets `layout="auto"` stop falling back to sparse on memory
     grounds, as the calculator already does.
2. **Keep slots at k(rcut + skin) + 8 for the benchmark**, where the
   structures collapse. On that deck, coordination within rcut grows by 8 or
   more within 250 steps. The rcut + skin list count is the natural bound:
   between rebuilds, no atom can have more neighbours within rcut than its
   list holds. So the safe gain today is chunking alone:
   - PACE: 1.27× at 131k atoms and 1.68× at 262k;
   - ACE: 2.75× at 65k and 7.7× at 131k.
3. **For a stable MD deck**, a thermalised crystal or liquid, k(rcut) plus
   about 20% would give the further 1.5×. The risk is an overflow, which
   aborts the run rather than truncating. That trade-off belongs to the user,
   not the default.
4. **Upstream, lammps-jax should test `!(d² <= cutsq)`**, so NaN pairs are
   dropped. With the current test, a NaN step is reported as an edge-capacity
   overflow at the full list size. That report is what misled Task 11.

## Gaps in the evidence

- The speed-up runs are 60 steps, too short for the benchmark structures to
  collapse. The chunked + rcut-slot column is therefore a best case. On the
  full benchmark deck, rcut-sized slots overflow for ACE Cantor large; see
  above.
- Only Cantor medium was measured.
- The synthetic inputs are slightly pessimistic for chunked bundles at large
  N: 134 against 117 ms in LAMMPS for PACE at 131k. Atom ordering is the
  likely reason; I did not investigate.
- I did not explain why XLA's gather-adjoint scatter slows per row beyond
  about 65k rows.

# Large linear fits

This page shows how to fit a linear model with a large basis or a large
dataset. It tells you which part of the fit takes the time, how to make a
long fit resumable, and how to spread the fit over the nodes of a cluster.

--8<-- "float64.md"

## What takes the time

A linear fit has two long stages:

1. **Training statistics.** ace-jax reads each batch of configurations,
   calculates its rows of the design matrix, and merges the rows into QR
   factors. The design matrix itself is never kept.
2. **MAP.** ace-jax finds the hyperparameters that give the highest evidence.
   Each step factorises a matrix of size L × L, where L is the number of basis
   functions.

The training statistics take most of the time when the dataset is large. The
force rows are the largest part, because each one needs the derivatives of all
L basis functions.

The progress file `progress.jsonl` in the output directory records the time of
each stage. The `objective` stage is the training statistics.

The table gives the measured rate of the training statistics for one dataset:
7800 configurations of 12 elements, in 4051 batches (L = 47,000).

| Hardware | Batches per hour for each node |
|---|---|
| one node, 128 cores (AMD EPYC) | approximately 55 |
| one node, 48 cores (Intel Xeon), one of 2 MPI ranks | approximately 148 |

The first batch also compiles the calculation. For this basis, the compilation
takes approximately 1.5 hours. With `--mpi`, the rate for each node does not
change, so N nodes take approximately 1/N of the time.

## Pack the configurations into batches

When the configurations have very different sizes, the default batches have
much padding. Packed batches have less padding.

1. Add `--batch-pack on` to `aj fit`.
2. Compare the padding in the log line `build_dataset: size-aware packing`.

`--batch-pack auto` (the default) packs only when the default batches have more
than twice the padding.

## Make a long fit resumable

A fit that stops, for example at a time limit, can continue from a checkpoint.

1. Add `--checkpoint DIR` to `aj fit`.
2. If the fit stops, run the same command again.

The fit then continues the training statistics from the last saved batch. If
the training statistics are complete, the fit starts the MAP from the best
hyperparameters found before.

`--checkpoint-every MIN` sets the time between two checkpoints. The default
is 45 minutes.

!!! warning "Checkpoints are large"
    A checkpoint holds 8 L² bytes for each quantity (energy, force and
    virial). At L = 20,000 this is approximately 9 GB, and at L = 47,000
    approximately 54 GB. A save can take several minutes. For a large basis,
    set `--checkpoint-every` to 120 minutes or more.

A checkpoint applies only to the same model, dataset and settings. ace-jax
does not load a checkpoint of a different fit.

## Spread a fit over nodes with MPI

The training statistics use the memory bandwidth of a node more than its
cores. Thus, more processes on one node give at most approximately 2 times
the speed. More nodes give approximately N times the speed for N nodes.

`aj fit --mpi` divides the batches between MPI processes (ranks). Each rank
calculates the QR factors of its batches. Then the ranks merge their factors
on rank 0, which does the MAP and writes all outputs.

!!! note "Only the training statistics are parallel"
    The MAP runs on rank 0 only. Its time does not decrease with more nodes.

### Install the MPI extra

mpi4py must use the MPI library of your cluster.

1. Load the MPI module of the cluster, for example `module load OpenMPI`.
2. Make sure that `mpicc` is on your `PATH`.
3. Install ace-jax with the `mpi` extra: `pip install "ace-jax[mpi]"`.

On a computer without an MPI module, `pip install mpich` gives an MPI library
for tests.

### Run the fit

Use one rank for each node. Give each rank all the cores of its node.

```bash
#!/usr/bin/env bash
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
module load GCC OpenMPI
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=48 OPENBLAS_NUM_THREADS=48
mpirun -np "$SLURM_NTASKS" --map-by ppr:1:node --bind-to none \
    -x JAX_PLATFORMS -x OMP_NUM_THREADS -x OPENBLAS_NUM_THREADS \
    aj fit --train train.xyz --test valid.xyz --order 3 --max-degree 4 \
    --batch-pack on --checkpoint ckpt --checkpoint-every 180 --mpi --out out
```

1. Change the paths, the basis specification and the number of nodes.
2. Submit the script with `sbatch`.
3. Read the progress of each rank in the error log (`qr stats rank ...`).

Rank 0 writes the log, `progress.jsonl` and all outputs. The other ranks write
only their progress lines to the error log.

!!! warning "Use a fast network"
    The ranks send their QR factors to rank 0 at the end of the training
    statistics: 8 L² bytes for each quantity. Use a cluster with a fast
    interconnect, for example InfiniBand.

With `--checkpoint`, each rank saves its own checkpoint in DIR. Run the same
command with the same number of ranks to continue.

## Check the result

An MPI fit gives the same model as a fit on one node, to round-off. To check
this on your cluster, fit a small dataset both ways and compare
`theta_map.json`.

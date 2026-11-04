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

The table gives the measured rates for one dataset: 3600 configurations of
8 elements in 1904 batches, with L = approximately 21,000.

| Stage | Hardware | Measured time |
|---|---|---|
| Training statistics | 4 nodes of 48 cores (Intel Xeon), one MPI rank each | approximately 1700 batches per hour for each node (17 minutes in total) |
| Merge of the 4 ranks, with checkpoints | the same 4 nodes | approximately 10 minutes |
| MAP, `--noise shared` | one node of 32 cores (mnf148) | approximately 40 seconds for each step |
| MAP, `--noise per-quantity` | one node of 32 cores (mnf148) | approximately 100 seconds for each step |

Reading the dataset took approximately 70 seconds. Compiling the calculation
took less than one minute. With `--mpi`, the rate for each node does not
change, so N nodes take approximately 1/N of the time.

The MAP usually needs tens of steps. Thus, for this dataset the MAP takes
approximately the same time as the training statistics. Its time does not
decrease with more nodes. See
[Run the MAP as a second job](#run-the-map-as-a-second-job).

The time of one MAP step increases as L³. At L = 47,000, one step takes
approximately 11 times longer than at L = 21,000.

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
    For a long MAP, use two jobs. See
    [Run the MAP as a second job](#run-the-map-as-a-second-job).

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

## Run the MAP as a second job

In one MPI job, the other nodes stay allocated but do no work while rank 0
does the MAP. Two jobs prevent this:

1. A job on N nodes calculates the training statistics and stops.
2. A job on one node reads the statistics from the checkpoint and does the MAP.

Do the steps that follow:

1. In the MPI job script, add `--stats-only` to `aj fit`. Keep `--checkpoint DIR`.
2. Write a second job script for one node. Use the same `aj fit` command, but
   remove `--mpi` and `--stats-only`. Do not use `mpirun`.
3. Submit the MPI job: `jobid=$(sbatch --parsable stats.sbatch)`.
4. Submit the second job after it: `sbatch --dependency=afterok:$jobid map.sbatch`.
5. Look for the log line `checkpoint: QR statistics complete` in the second job.

```bash
#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --time=48:00:00
export JAX_PLATFORMS=cpu OMP_NUM_THREADS=48 OPENBLAS_NUM_THREADS=48
aj fit --train train.xyz --test valid.xyz --order 3 --max-degree 4 \
    --batch-pack on --checkpoint ckpt --out out
```

With `--stats-only`, rank 0 merges the factors of all ranks and saves them as
one checkpoint file. The second job reads this file. It does not use the
checkpoint files of the ranks.

The second job must use the same model, dataset and settings as the first job.
If a setting is different, the second job does not find the checkpoint and
calculates the training statistics again on one node.

!!! warning "The MAP needs much memory"
    The MAP needs approximately 90 L² bytes of memory. This is approximately
    40 GB at L = 21,000 and approximately 200 GB at L = 47,000. For a large
    basis, use a large-memory node for the second job.

!!! note "Disk space for the checkpoints"
    With `--mpi`, the checkpoint directory holds the checkpoint of each rank
    and the merged checkpoint. Each one is 24 L² bytes. When the second job
    has started, you can delete the files of the ranks
    (`qrstats-*-r*of*.npz`).

If the second job stops at its time limit, submit it again. It continues the
MAP from the best hyperparameters found before.

## Check the result

An MPI fit gives the same model as a fit on one node, to round-off. To check
this on your cluster, fit a small dataset both ways and compare
`theta_map.json`.

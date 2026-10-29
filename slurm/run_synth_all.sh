#!/bin/bash

#SBATCH --job-name=fairsketch_synth_all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16GB
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --account=torch_pr_594_tandon_priority
#SBATCH --output=/scratch/rm6609/fair-sketch-experiments/logs/%x_%A_%a.log
#SBATCH --mail-user=rm6609@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# All synthetic experiments (Sections 4 and 5, regression and classification). One array task per driver, each running
# its full default set of experiments in one process (so the random stream matches a plain local run of the driver):
#
#   task 0 s4_synthetic_regression      synth_m, synth_kappa, synth_gap, synth_lev, synth_spike, bounds_table  -> results/reg_s4_synth
#        1 s5_synthetic_regression      table, m, kappa, gap, lev, pilot, beta                                 -> results/reg_s5_synth
#        2 s4_synthetic_classification  table + sweeps m, gap, minority, steps                                 -> results/class_s4_synth
#        3 s5_synthetic_classification  table + sweeps m, steps, minority, mis                                 -> results/class_s5_synth
#
# Subsets:  sbatch --array=0,1 slurm/run_synth_all.sh   (regression only)
#           EXTRA_ARGS="--quick" sbatch --array=0-3 slurm/run_synth_all.sh   (smoke test: few reps, small n, fewer sweep points)
# Before the first run:  mkdir -p /scratch/rm6609/fair-sketch-experiments/logs

REPO="/scratch/rm6609/fair-sketch-experiments"
SEED=0
REG_REPS=200
CLASS_REPS=60
EXTRA_ARGS="${EXTRA_ARGS:-}"      # passed to every driver, e.g. "--quick" or "--latex"

DRIVERS=(s4_synthetic_regression s5_synthetic_regression s4_synthetic_classification s5_synthetic_classification)
OUTS=(reg_s4_synth reg_s5_synth class_s4_synth class_s5_synth)

TASK=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK < 0 || TASK >= ${#DRIVERS[@]} )); then
    echo "ERROR: array task $TASK out of range 0-$(( ${#DRIVERS[@]} - 1 ))" >&2; exit 1
fi
DRIVER=${DRIVERS[$TASK]}
OUT_DIR="${REPO}/results/${OUTS[$TASK]}"

# regression drivers default to --reps 100 (fs_common); classification to 60 (lib_classification); set both explicitly
case "$DRIVER" in
    *_regression)     EXP_ARGS="--reps $REG_REPS --exp all" ;;
    *_classification) EXP_ARGS="--reps $CLASS_REPS" ;;
esac

# snapshot the code so edits made while the job is queued or running do not change it
SNAPSHOT_ROOT="${REPO}/.job_snapshots"
SNAPSHOT_DIR="${SNAPSHOT_ROOT}/${SLURM_ARRAY_JOB_ID:-$$}_synth_${TASK}_$(date +%s)"
mkdir -p "$SNAPSHOT_DIR" || { echo "ERROR: failed to create snapshot dir $SNAPSHOT_DIR" >&2; exit 1; }
rsync -a --exclude='__pycache__' --exclude='.git' --exclude='.job_snapshots' --exclude='results' --exclude='logs' \
      --exclude='hmda_cache' --exclude='data' --exclude='local_data' --exclude='old' --exclude='*.csv' --exclude='*.parquet' \
      "${REPO}/" "${SNAPSHOT_DIR}/" \
    || { echo "ERROR: failed to snapshot the code into $SNAPSHOT_DIR" >&2; exit 1; }
mkdir -p "$OUT_DIR"

CMD="python3 ${SNAPSHOT_DIR}/synthetic/${DRIVER}.py --out $OUT_DIR --seed $SEED $EXP_ARGS $EXTRA_ARGS"

echo "========================================"
echo "Job Start: $(date '+%Y-%m-%d_%H-%M-%S') | array task $TASK: $DRIVER"
echo "Code snapshot: $SNAPSHOT_DIR"
echo "Command: $CMD"
echo "========================================"

OVERLAY="/scratch/rm6609/research/overlay-persistent-manual-2.ext3"

singularity exec --fakeroot --overlay "$OVERLAY:ro" \
/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
/bin/bash -c "
    source /ext3/miniconda3/bin/activate research
    export HOME=/ext3/conda_home
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
    # the overlay is read-only: send the matplotlib cache somewhere writable, one per task
    export MPLCONFIGDIR=\${TMPDIR:-/tmp}/mpl_${SLURM_ARRAY_JOB_ID:-$$}_synth_${TASK}
    cd $REPO
    $CMD
"
JOB_EXIT=$?

if [[ "$SNAPSHOT_DIR" == "$SNAPSHOT_ROOT"/* ]]; then
    rm -rf -- "$SNAPSHOT_DIR"
fi

echo "Job End: $(date '+%Y-%m-%d_%H-%M-%S') | exit $JOB_EXIT"
exit "$JOB_EXIT"

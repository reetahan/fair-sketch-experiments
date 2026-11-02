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

# tasks: 0 s4 regression, 1 s5 regression, 2 s4 classification, 3 s5 classification
# smoke test: EXTRA_ARGS="--quick" sbatch slurm/run_synth_all.sh

REPO="/scratch/rm6609/fair-sketch-experiments"
SEED=0
REG_REPS=200
CLASS_REPS=60
EXTRA_ARGS="${EXTRA_ARGS:-}"

DRIVERS=(s4_synthetic_regression s5_synthetic_regression s4_synthetic_classification s5_synthetic_classification)
OUTS=(reg_s4_synth reg_s5_synth class_s4_synth class_s5_synth)

TASK=${SLURM_ARRAY_TASK_ID:-0}
if (( TASK < 0 || TASK >= ${#DRIVERS[@]} )); then
    echo "ERROR: array task $TASK out of range 0-$(( ${#DRIVERS[@]} - 1 ))" >&2; exit 1
fi
DRIVER=${DRIVERS[$TASK]}
OUT_DIR="${REPO}/results/${OUTS[$TASK]}"

case "$DRIVER" in
    *_regression)     EXP_ARGS="--reps $REG_REPS --exp all" ;;
    *_classification) EXP_ARGS="--reps $CLASS_REPS" ;;
esac

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

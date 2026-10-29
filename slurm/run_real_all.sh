#!/bin/bash

#SBATCH --job-name=fairsketch_real_all
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=08:00:00
#SBATCH --array=0-19
#SBATCH --account=torch_pr_594_tandon_priority
#SBATCH --output=/scratch/rm6609/fair-sketch-experiments/logs/%x_%A_%a.log
#SBATCH --mail-user=rm6609@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# All real-data experiments (Sections 4 and 5, regression and classification) on all 5 real datasets.
# One array task per (driver, dataset):  task = 5 * driver + dataset
#
#   driver   0 s4_real_regression      bounds table, vary m (size/direction)            -> results/reg_s4_real
#            1 s5_real_regression      OLS/vanilla/oracle/proxy table, vary m            -> results/reg_s5_real
#            2 s4_real_classification  approaches 1-3 table, vary m, vary Newton steps   -> results/class_s4_real
#            3 s5_real_classification  vanilla/oracle/proxy table, vary m, vary steps    -> results/class_s5_real
#   dataset  0 LSAC   1 ACS CA 2018   2 HMDA CA 2018   3 Diabetes 130-US   4 ENEM 2020
#
# The LSAC regression tasks (0 and 5) also run the ablations, which are defined on LSAC only (as before).
# Subsets:  sbatch --array=0-4 slurm/run_real_all.sbatch       (all datasets, s4 regression)
#           sbatch --array=3,8,13,18 slurm/run_real_all.sbatch (Diabetes, all four drivers)
# Before the first run:  mkdir -p /scratch/rm6609/fair-sketch-experiments/logs

REPO="/scratch/rm6609/fair-sketch-experiments"
HMDA_PATH="${REPO}/hmda_cache/hmda_CA_2018.csv"      # CA 2018 CSV from the FFIEC data browser (the API returned 403); "" = use the API
ENEM_PATH="${REPO}/data/MICRODADOS_ENEM_2020.csv"    # INEP microdata; required for the ENEM tasks
SEED=0
REG_REPS=200
CLASS_REPS=60

DRIVERS=(s4_real_regression s5_real_regression s4_real_classification s5_real_classification)
OUTS=(reg_s4_real reg_s5_real class_s4_real class_s5_real)
DATASETS=(law acs hmda diab enem)

TASK=${SLURM_ARRAY_TASK_ID:-0}
DRIVER_IDX=$(( TASK / 5 ))
DATA_IDX=$(( TASK % 5 ))
DRIVER=${DRIVERS[$DRIVER_IDX]}
DATASET=${DATASETS[$DATA_IDX]}
OUT_DIR="${REPO}/results/${OUTS[$DRIVER_IDX]}"

# dataset selection: every driver loads LSAC, ACS and HMDA by default, Diabetes and ENEM only when asked
case "$DATASET" in
    law)  DATA_ARGS="--no_acs --no_hmda" ;;
    acs)  DATA_ARGS="--no_law --no_hmda" ;;
    hmda) DATA_ARGS="--no_law --no_acs"
          [[ -n "$HMDA_PATH" ]] && DATA_ARGS="$DATA_ARGS --hmda_path $HMDA_PATH" ;;
    diab) DATA_ARGS="--no_law --no_acs --no_hmda --diab" ;;
    enem) DATA_ARGS="--no_law --no_acs --no_hmda --enem_path $ENEM_PATH"
          [[ -f "$ENEM_PATH" ]] || { echo "ERROR: ENEM file not found: $ENEM_PATH" >&2; exit 1; } ;;
esac
if [[ "$DATASET" == "hmda" && -n "$HMDA_PATH" && ! -f "$HMDA_PATH" ]]; then
    echo "ERROR: HMDA file not found: $HMDA_PATH (set HMDA_PATH=\"\" to use the API)" >&2; exit 1
fi

# which experiments: regression drivers take --exp; ablations only on LSAC (they would overwrite each other otherwise)
case "$DRIVER" in
    s4_real_regression) EXP_ARGS="--reps $REG_REPS --exp $([[ $DATASET == law ]] && echo all || echo bounds_table,real_m)" ;;
    s5_real_regression) EXP_ARGS="--reps $REG_REPS --exp $([[ $DATASET == law ]] && echo all || echo table,m)" ;;
    *)                  EXP_ARGS="--reps $CLASS_REPS" ;;
esac

# snapshot the code so edits made while the job is queued or running do not change it
SNAPSHOT_ROOT="${REPO}/.job_snapshots"
SNAPSHOT_DIR="${SNAPSHOT_ROOT}/${SLURM_ARRAY_JOB_ID:-$$}_${TASK}_$(date +%s)"
mkdir -p "$SNAPSHOT_DIR" || { echo "ERROR: failed to create snapshot dir $SNAPSHOT_DIR" >&2; exit 1; }
rsync -a --exclude='__pycache__' --exclude='.git' --exclude='.job_snapshots' --exclude='results' --exclude='logs' \
      --exclude='hmda_cache' --exclude='data' --exclude='*.csv' --exclude='*.parquet' "${REPO}/" "${SNAPSHOT_DIR}/" \
    || { echo "ERROR: failed to snapshot the code into $SNAPSHOT_DIR" >&2; exit 1; }
mkdir -p "$OUT_DIR"

CMD="python3 ${SNAPSHOT_DIR}/real/${DRIVER}.py --out $OUT_DIR --seed $SEED $EXP_ARGS $DATA_ARGS"

echo "========================================"
echo "Job Start: $(date '+%Y-%m-%d_%H-%M-%S') | array task $TASK: $DRIVER on $DATASET"
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
    # the overlay is read-only: send the OpenML (Diabetes) and matplotlib caches somewhere writable, one per task
    export SCIKIT_LEARN_DATA=\${TMPDIR:-/tmp}/sklearn_${SLURM_ARRAY_JOB_ID:-$$}_${TASK}
    export MPLCONFIGDIR=\${TMPDIR:-/tmp}/mpl_${SLURM_ARRAY_JOB_ID:-$$}_${TASK}
    cd $REPO
    $CMD
"
JOB_EXIT=$?

if [[ "$SNAPSHOT_DIR" == "$SNAPSHOT_ROOT"/* ]]; then
    rm -rf -- "$SNAPSHOT_DIR"
fi

echo "Job End: $(date '+%Y-%m-%d_%H-%M-%S') | exit $JOB_EXIT"
exit "$JOB_EXIT"

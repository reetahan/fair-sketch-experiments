#!/bin/bash

#SBATCH --job-name=fairsketch_mlp
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --array=0-11
#SBATCH --account=torch_pr_594_tandon_priority
#SBATCH --output=/scratch/rm6609/fair-sketch-experiments/logs/%x_%A_%a.log
#SBATCH --mail-user=rm6609@nyu.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# task = 6 * kind + dataset
# kind: 0 regression, 1 classification
# dataset: 0 LSAC, 1 ACS, 2 HMDA, 3 Diabetes, 4 ENEM, 5 synthetic

REPO="/scratch/rm6609/fair-sketch-experiments"
TORCH_DEPS="/scratch/rm6609/research/pydeps_torch"
HMDA_PATH="${REPO}/local_data/hmda_CA_2018.csv"
ENEM_PATH="${REPO}/local_data/DADOS/MICRODADOS_ENEM_2020.csv"
DIAB_PATH="${REPO}/local_data/diabetes_hospital.csv"
OUT_DIR="${REPO}/results/mlp_ext"
SEED=0
EXTRA_ARGS="${EXTRA_ARGS:-}"

KINDS=(reg cls)
DATASETS=(law acs hmda diab enem synth)

TASK=${SLURM_ARRAY_TASK_ID:-0}
KIND=${KINDS[$(( TASK / 6 ))]}
DATASET=${DATASETS[$(( TASK % 6 ))]}

case "$DATASET" in
    law)   DATA_ARGS="--no_acs --no_hmda" ;;
    acs)   DATA_ARGS="--no_law --no_hmda" ;;
    hmda)  DATA_ARGS="--no_law --no_acs --hmda_path $HMDA_PATH" ;;
    diab)  DATA_ARGS="--no_law --no_acs --no_hmda --diab --diab_path $DIAB_PATH" ;;
    enem)  DATA_ARGS="--no_law --no_acs --no_hmda --enem_path $ENEM_PATH" ;;
    synth) DATA_ARGS="--synthetic" ;;
esac
[[ "$KIND" == "cls" ]] && DATA_ARGS="$DATA_ARGS --acs_n 50000 --hmda_n 50000"

SNAPSHOT_ROOT="${REPO}/.job_snapshots"
SNAPSHOT_DIR="${SNAPSHOT_ROOT}/${SLURM_ARRAY_JOB_ID:-$$}_mlp_${TASK}_$(date +%s)"
mkdir -p "$SNAPSHOT_DIR" || { echo "ERROR: failed to create snapshot dir $SNAPSHOT_DIR" >&2; exit 1; }
rsync -a --exclude='__pycache__' --exclude='.git' --exclude='.job_snapshots' --exclude='results' --exclude='logs' \
      --exclude='hmda_cache' --exclude='data' --exclude='local_data' --exclude='old' --exclude='*.csv' --exclude='*.parquet' \
      "${REPO}/" "${SNAPSHOT_DIR}/" \
    || { echo "ERROR: failed to snapshot the code into $SNAPSHOT_DIR" >&2; exit 1; }
mkdir -p "$OUT_DIR"

CMD="python3 -u ${SNAPSHOT_DIR}/extensions/mlp_fairness.py --task $KIND --out $OUT_DIR --seed $SEED $DATA_ARGS $EXTRA_ARGS"

echo "========================================"
echo "Job Start: $(date '+%Y-%m-%d_%H-%M-%S') | array task $TASK: $KIND on $DATASET"
echo "Command: $CMD"
echo "========================================"

OVERLAY="/scratch/rm6609/research/overlay-persistent-manual-2.ext3"

singularity exec --fakeroot --overlay "$OVERLAY:ro" \
/share/apps/images/cuda13.0.1-cudnn9.13.0-ubuntu-24.04.3.sif \
/bin/bash -c "
    source /ext3/miniconda3/bin/activate research
    export HOME=/ext3/conda_home
    export PYTHONPATH=$TORCH_DEPS
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
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

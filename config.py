import re
from pathlib import Path


##########################################################################
# Set file path for log file and logging level                           #
##########################################################################

LOG_FILE_PATH = "outputs/logs/guardian_qc.log"
LOG_LEVEL = "INFO"

##########################################################################
# Set file paths and regex patterns for run folders and qc_summary files #
# For pipeline/ingest.py                                                 #
##########################################################################

# Set all paths and regex patterns for run folders and qc_summary files
# Path allows you to manipulate windows paths on Unix machines
ROOT_FILE_PATH = Path("/mnt/dxstream/outputs")
NOVASEQX_PATH = Path("/mnt/dxstream/runs/NovaSeqX")
NOVASEQ6000_PATH = Path("/mnt/dxstream/runs/NovaSeq")

# Define regex patterns for run folders and qc_summary files
RUN_PATTERN_NOVASEQX = re.compile(r"^\d{6,8}_LH00537_\d{4}_[A-Z0-9]{10}$")
RUN_PATTERN_NOVASEQ6000 = re.compile(r"^\d{6,8}_A01184_\d{4}_[A-Z0-9]{10}$")

# Define regex pattern for subfolders (7-digit numbers) - the worklist folders
RUN_PATTERN_SUB = re.compile(r"^[0-9]{7}$")

# Define regex patterns for qc_summary files for different cancer types and versions
QC_SUMMARY_PATTERN_ST_V3 = re.compile(r"^[0-9]{7}\.RMH200STv3\.qc_summary\.tsv$")
QC_SUMMARY_PATTERN_ST = re.compile(r"^[0-9]{7}\.RMH200ST\.qc_summary\.tsv$")
QC_SUMMARY_PATTERN_HAEM_V2 = re.compile(r"^[0-9]{7}\.RMHhaemV2\.qc_summary\.tsv$")
QC_SUMMARY_PATTERN_HAEM_V3 = re.compile(r"^[0-9]{7}\.RMHhaemV3\.qc_summary\.tsv$")

# sample sheet pattern to match SampleSheet.csv files
SAMPLE_SHEET_PATTERN = re.compile(r"^SampleSheet\.csv$")

# Specify output file locations for the QC summary files
QC_FILE_PATHS = "data/raw/qc_summary_file_paths.csv"

# Set output file location for files that pass run-level QC filtering and have all information for downstream analysis
FILTERED_RUN_QC_DATA = "data/processed/filtered_by_run_metrics(1).csv"

# Set file path to summary_qc_metrics.csv file
SUMMARY_QC_METRICS = "data/processed/summary_qc_metrics(2).csv"
SUMMARY_QC_METRICS_CLEANED = "data/processed/cleaned_summary_qc_metrics(3).csv"

# Set variables for preprocessing 
# Define the model features 
MODEL_FEATURES = [
    "bcftools_ts",
    "bcftools_tv",
    "bcftools_tstv",
    "bcftools_variants",
    "bcftools_snvs",
    "bcftools_indels",
    "picard_mode_insert",
    "picard_mean_insert",
    "picard_median_insert",
    "picard_mad_insert",
    "picard_total_reads",
    "picard_pf_reads",
    "picard_pf_q30_bases",
    "picard_read_length",
    "picard_at_dropout",
    "picard_gc_dropout",
    "picard_fold_enrichment",
    "picard_fold80",
    "picard_mean_target_coverage",
    "picard_median_target_coverage",
    "picard_target_bases_20x",
    "picard_target_bases_30x",
    "picard_target_bases_50x",
    "picard_target_bases_100x",
    "fastqc_duplication_rate",
    "fastp_duplication_rate"
]

CATERGORICAL_COLUMNS = ['fastqc_basic_status']
METADATA_COLUMNS = ['sample_name', 'cancer_type', 'sequencer']
STRATIFY_COLUMNS = ['cancer_type', 'sequencer']

# Set output directory for preprocessing files 
PREPROCESSING_OUTDIR = 'models/preprocessing/'
UNSUPERVISED_MODELS_DIR = 'models/unsupervised/'
PREPROCESSING_PLOT_DIR = 'outputs/graphs/preprocessing/'


import re
from pathlib import Path

##########################################################################
# Set file paths and regex patterns for run folders and qc_summary files #
# For utils_cancer.py                                                    #
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
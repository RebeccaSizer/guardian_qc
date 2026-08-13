import argparse
import config

from pipeline.ingest import get_run_file_paths, get_qc_summary_file_path, get_run_sample_file_paths, filter_df, filter_run_qc, sample_level_qc
from utils.logger import logging

df_sample = get_qc_summary_file_path()
df_run = get_run_file_paths()

df = get_run_sample_file_paths(df_run, df_sample)
df.to_csv(config.QC_FILE_PATHS, sep="\t", index=False)

# Filter the data
df = filter_df(df)
df = filter_run_qc(df)
df.to_csv(config.FILTERED_RUN_QC_DATA, sep="\t", index=False)

# Extract summary QC metrics
summary_qc_metrics_df = sample_level_qc(df)
summary_qc_metrics_df.to_csv(config.SUMMARY_QC_METRICS, sep = "\t", index = False)
print(summary_qc_metrics_df)
import argparse
import config

from pipeline.ingest import get_run_file_paths, get_qc_summary_file_path, merge_run_and_qc_data, filter_df, filter_run_qc, sample_level_qc
from utils.logger import logging

#######################################################
# Ingest and process QC summary and run summary files # 
#######################################################

df_sample = get_qc_summary_file_path()
df_run = get_run_file_paths()
df = merge_run_and_qc_data(df_run, df_sample)
df.to_csv(config.QC_FILE_PATHS, sep="\t", index=False)
df = filter_df(df)
df = filter_run_qc(df)
df.to_csv(config.FILTERED_RUN_QC_DATA, sep="\t", index=False)
summary_qc_metrics_df = sample_level_qc(df)
summary_qc_metrics_df.to_csv(config.SUMMARY_QC_METRICS, sep = "\t", index = False)


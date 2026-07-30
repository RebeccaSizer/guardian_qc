""" 
This module contains functions for preprocessing machine learning data.
"""

import pandas as pd
from tools.utils.logger import logging

def sample_level_qc(df):
    """
    Function to perform sample-level QC on sequencing data.
    This function is a placeholder for the actual implementation of sample-level QC.
    
    params:
        None
    
    output:
        df: pd.DataFrame containing the summarized sample-level QC metrics for each sample.
    """
    summary_qc_metrics = []

    for _, row in df.iterrows():

        if row["pass_run_qc"] == "Yes":

            summary_qc_file_path = row["qc_file_path"]
            sample_qc = extract_values_from_qc_summary(summary_qc_file_path)

            for sample in sample_qc:
                summary_qc_metrics.append(sample)

    summary_qc_metrics_df = pd.DataFrame(summary_qc_metrics)

    return summary_qc_metrics_df


def extract_values_from_qc_summary(file_path):
    """
    Function to extract values from the QC summary file.
    This function is a placeholder for the actual implementation of extracting values from the QC summary file.
    
    params:
        file_path: str, path to the QC summary file
    
    output:
        qc_values: dict, containing the extracted values from the QC summary file.
    """

    sample_qc = []

    try:

        df = pd.read_csv(file_path, sep = "\t")

        for _, row in df.iterrows():

            logging.info(f"Extracting QC metrics for sample: {row['sample_name']}")

            sample_qc.append({
                "sample_name": row["sample_name"],
                "bcftools_ts": row["bcftools_ts"],
                "bcftools_tv": row["bcftools_tv"],
                "bcftools_tstv": row["bcftools_tstv"],
                "bcftools_variants": row["bcftools_variants"],
                "bcftools_snvs": row["bcftools_SNVs"],
                "bcftools_indels": row["bcftools_indels"],
                "picard_mode_insert": row["picard_mode_insert"],
                "picard_mean_insert": row["picard_mean_insert"],
                "picard_median_insert": row["picard_median_insert"],
                "picard_mad_insert": row["picard_mad_insert"],
                "picard_total_reads": row["picard_total_reads"],
                "picard_pf_reads": row["picard_pf_reads"],
                "picard_pf_q30_bases": row["picard_pf_q30_bases"],
                "picard_read_length": row["picard_read_length"],
                "picard_at_dropout": row["picard_at_dropout"],
                "picards_gc_dropout": row["picard_gc_dropout"],
                "picard_fold_enrichment": row["picard_fold_enrichment"],
                "picard_fold80": row["picard_fold80"],
                "picard_mean_target_coverage": row["picard_mean_target_coverage"],
                "picard_median_target_coverage": row["picard_median_target_coverage"],
                "picard_target_bases_20x": row["picard_target_bases_20x"],
                "picard_target_bases_30x": row["picard_target_bases_30x"],
                "picard_target_bases_50x": row["picard_target_bases_50x"],
                "picard_target_bases_100x": row["picard_target_bases_100x"],
                "fastqc_duplication_rate": row["fastqc_duplication_rate"],
                "fastqc_basic_status": row["fastqc_basic_status"],
                "fastp_duplication_rate": row["fastp_duplication_rate"]
            })

        return sample_qc

    except Exception as e:
        logging.error(f"Error reading QC summary file {file_path}: {e}")
            
        return sample_qc


if __name__ == "__main__":
    samples = pd.read_csv("outputs/filtered_run_metrics.csv", sep = "\t")
    logging.info("Starting sample-level QC processing...")
    summary_qc_metrics_df = sample_level_qc(samples)
    summary_qc_metrics_df.to_csv("outputs/summary_qc_metrics.csv", sep = "\t", index = False)

""" 
 This script processes and filters the run metrics so that only
 samples that pass run level metrics are kept"""
import logging

from tools.modules.inter_op import inter_op_qc
import pandas as pd
from pathlib import Path

def filter_run_qc(df):

    pass_list = []

    for _, row in df.iterrows():

        run_folder_path = row["run_qual_filepath"]
        run_folder_name = Path(run_folder_path).name

        run_df = inter_op_qc(run_folder_path)

        # Select the correct row of the InterOp summary
        lane = row["lane"]

        if lane == [1]:
            run_columns = run_df.loc[0]
        elif lane == [2]:
            run_columns = run_df.loc[1]
        elif lane == [1, 2]:
            run_columns = run_df.loc[2]
        elif lane is None:
            logging.warning(f"Lane information is missing for run {run_folder_name}. Skipping this run.")
            continue
        else:
            raise ValueError(f"Unexpected lane value: {lane}")

        q30 = run_columns["Percent Q30"]
        error_rate = run_columns["Error Rate"]

        if q30 >= 80 and error_rate <= 2:
            status = "Yes"
        elif q30 < 80 and error_rate <= 2:
            status = "Percent Q30 < 80"
        elif q30 >= 80 and error_rate > 2:
            status = "Error rate > 2"
        else:
            status = "Percent Q30 < 80 AND Error rate > 2"

        pass_list.append({
            "seq_run_number": run_folder_name,
            "pass_run_qc": status
        })

    return pd.DataFrame(pass_list)


if __name__=="__main__":

    df = pd.DataFrame({
    "seq_run_number": [
        "20260716_LH00537_0196_A22KWG7LT1",
        "20260716_LH00537_0196_A22KWG7LT1",
        "20260716_LH00537_0196_A22KWG7LT1",
        "20260716_LH00537_0196_A22KWG7LT1",
        "20260716_LH00537_0196_A22KWG7LT1",
        "20250912_LH00537_0019_A22FVH3LT1",
        "20250912_LH00537_0019_A22FVH3LT1",
        "20250912_LH00537_0019_A22FVH3LT1",
        "20250912_LH00537_0019_A22FVH3LT1",
        "20250912_LH00537_0019_A22FVH3LT1",
    ],
    "run_qual_filepath": [
        "/mnt/dxstream/runs/NovaSeqX/20260716_LH00537_0196_A22KWG7LT1",
        "/mnt/dxstream/runs/NovaSeqX/20260716_LH00537_0196_A22KWG7LT1",
        "/mnt/dxstream/runs/NovaSeqX/20260716_LH00537_0196_A22KWG7LT1",
        "/mnt/dxstream/runs/NovaSeqX/20260716_LH00537_0196_A22KWG7LT1",
        "/mnt/dxstream/runs/NovaSeqX/20260716_LH00537_0196_A22KWG7LT1",
        "/mnt/dxstream/runs/NovaSeqX/20250912_LH00537_0019_A22FVH3LT1",
        "/mnt/dxstream/runs/NovaSeqX/20250912_LH00537_0019_A22FVH3LT1",
        "/mnt/dxstream/runs/NovaSeqX/20250912_LH00537_0019_A22FVH3LT1",
        "/mnt/dxstream/runs/NovaSeqX/20250912_LH00537_0019_A22FVH3LT1",
        "/mnt/dxstream/runs/NovaSeqX/20250912_LH00537_0019_A22FVH3LT1",
    ],
    "worklist": [
        2617408,
        2617462,
        2617460,
        2617326,
        2617323,
        2520555,
        2521117,
        2520658,
        2520553,
        2520653,
    ],
    "qc_file_path": [
        "/mnt/dxstream/outputs/20260716_LH00537_0196_A22KWG7LT1/2617408/2617408.RMH200STv3.qc_summary.tsv",
        "/mnt/dxstream/outputs/20260716_LH00537_0196_A22KWG7LT1/2617462/2617462.RMHhaemV3.qc_summary.tsv",
        "/mnt/dxstream/outputs/20260716_LH00537_0196_A22KWG7LT1/2617460/2617460.RMH200STv3.qc_summary.tsv",
        "/mnt/dxstream/outputs/20260716_LH00537_0196_A22KWG7LT1/2617326/2617326.RMH200STv3.qc_summary.tsv",
        "/mnt/dxstream/outputs/20260716_LH00537_0196_A22KWG7LT1/2617323/2617323.RMHhaemV3.qc_summary.tsv",
        "/mnt/dxstream/outputs/20250912_LH00537_0019_A22FVH3LT1/2520555/2520555.RMHhaemV2.qc_summary.tsv",
        "/mnt/dxstream/outputs/20250912_LH00537_0019_A22FVH3LT1/2521117/2521117.RMHhaemV2.qc_summary.tsv",
        "/mnt/dxstream/outputs/20250912_LH00537_0019_A22FVH3LT1/2520658/2520658.RMHhaemV2.qc_summary.tsv",
        "/mnt/dxstream/outputs/20250912_LH00537_0019_A22FVH3LT1/2520553/2520553.RMH200ST.qc_summary.tsv",
        "/mnt/dxstream/outputs/20250912_LH00537_0019_A22FVH3LT1/2520653/2520653.RMH200ST.qc_summary.tsv",
    ],
    "sequencer": [
        "novaseqx",
        "novaseqx",
        "novaseqx",
        "novaseqx",
        "novaseqx",
        "novaseqx",
        "novaseqx",
        "novaseqx",
        "novaseqx",
        "novaseqx",
    ],
    "cancer_type": [
        "solid_tumour_v3",
        "haem_v3",
        "solid_tumour_v3",
        "solid_tumour_v3",
        "haem_v3",
        "haem_v2",
        "haem_v2",
        "haem_v2",
        "solid_tumour",
        "solid_tumour",
    ],
    "file_status": [
        "Yes",
        "Yes",
        "Yes",
        "Yes",
        "Yes",
        "Yes",
        "Yes",
        "Yes",
        "Yes",
        "Yes",
    ]
    })

    df_run_pass = filter_run_qc(df)
    print(df_run_pass)
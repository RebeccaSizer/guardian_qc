""" 
This script processes and filters the run metrics so that only
samples that pass run level metrics are kept
"""

import pandas as pd
from pathlib import Path

from tools.utils.logger import logging
from tools.modules.inter_op import inter_op_qc

def filter_run_qc(df):
    """
    Filter the dataframe to include only runs that pass run level QC metrics.
    The function checks the 'Percent Q30' and 'Error Rate' for each run and
    determines if the run passes QC based on the following criteria:

    - If 'Percent Q30' >= 80 and 'Error Rate' <= 2
    - If 'Percent Q30' < 80 and 'Error Rate' is NaN
    
    If a run does not meet these criteria, it is marked as failing QC.
    
    params:
        df (pd.DataFrame): DataFrame containing run metrics and QC information.
    
    output:
        pd.DataFrame: Filtered DataFrame containing only runs that pass QC.
        
    """

    pass_list = []

    for _, row in df.drop_duplicates("seq_run_number").iterrows():
        print(df.columns.tolist())

        run_folder_path = row["run_qual_filepath"]
        run_folder_name = Path(run_folder_path).name

        run_df = inter_op_qc(run_folder_path)

        if pd.notna(row["sequencer"]) and "novaseqx" in row["sequencer"].lower():
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

        elif pd.notna(row["sequencer"]) and "novaseq6000" in row["sequencer"].lower():
            index = run_df.index[run_df["Lane"] == "Full Run"][0]
            run_columns = run_df.loc[index]

        else:
            raise ValueError(f"Unexpected sequencer type: {row['sequencer']}")

        q30 = run_columns["Percent Q30"]
        error_rate = run_columns["Error Rate"]

        if q30 >= 80 and error_rate <= 2 or q30 < 80 and error_rate == "NaN":
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

        df_pass = df.merge(pd.DataFrame(pass_list), 
                           how="left", 
                           left_on="seq_run_number", 
                           right_on="seq_run_number")

        df_pass.to_csv("outputs/filtered_run_metrics.csv", sep="\t", index=False)

    return df_pass 


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
    "lane": [
        [1],
        [1],
        [1],
        [1],
        [1],
        [1, 2],
        [1, 2],
        [1, 2],
        [1, 2],
        [1, 2],
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
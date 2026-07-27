""" 
 This script processes and filters the run metrics so that only
 samples that pass run level metrics are kept"""
from tools.modules.inter_op import run_qc_summary
import pandas as pd
from pathlib import Path

def filter_run_qc(df):

    pass_list = []
    
    for run_folder_path in df["run_qual_filepath"]:
        run_df = run_qc_summary(run_folder_path)
        run_folder_name = Path(run_folder_path).name

        run_both_columns = run_df.loc[2]

        if run_both_columns["Percent Q30"] >= 80 and run_both_columns["Error Rate"] <= 2:

            pass_list.append({
                "seq_run_number" : run_folder_name,
                "pass_run_qc" : "Yes"
            })

        elif run_both_columns["Percent Q30"] < 80 and run_both_columns["Error Rate"] <= 2:

            pass_list.append({
                            "seq_run_number" : run_folder_name,
                            "pass_run_qc" : "Percent Q30 < 80"
                        })

        elif run_both_columns["Percent Q30"] >= 80 and run_both_columns["Error Rate"] > 2:
        
                    pass_list.append({
                                    "seq_run_number" : run_folder_name,
                                    "pass_run_qc" : "Error rate > 2"
                                })

        else:
            pass_list.append({
                            "seq_run_number" : run_folder_name,
                            "pass_run_qc" : "Percent Q30 < 80 AND Error rate > 2"
                        })

    pass_filtering_df = pd.DataFrame(pass_list, columns=["seq_run_number", "pass_run_qc"])

    return pass_filtering_df



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
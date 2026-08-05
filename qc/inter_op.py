"""Gather run-level QC with lane breakdown from InterOp folder"""

# Import packages needed for tool 
import argparse
import os.path
from interop import py_interop_run_metrics, py_interop_summary
import numpy as np
import pandas as pd
from utils.logger import logger

# The following code was written by MW and edited by RS.
def inter_op_qc(run_path):
    """
    Function to extract run-level QC metrics from the InterOp folder of a sequencing run.
    The function reads the InterOp files, summarizes the run metrics, and returns a DataFrame
    containing the following metrics for each lane and the full run:
    
    - Lane number
    - Yield in gigabases (Gb)
    - Percent of bases with quality score >= 30 (Percent Q30)
    - Percent of clusters passing filter (Percent PF)
    - Error rate (Error Rate)

    params:
        run_path (str): Path to the InterOp folder of the sequencing run.

    output:
        pd.DataFrame: DataFrame containing the summarized run-level QC metrics for each lane and the full run.

    """
    
    # get full path for run folder
    logger.info(f"Accessing run forlder data at path : {run_path}")

    # Load InterOp
    run_metrics = py_interop_run_metrics.run_metrics()
    run_metrics.read(run_path)
    summary = py_interop_summary.run_summary()
    py_interop_summary.summarize_run_metrics(run_metrics, summary)

    # Explore the data
    #print(run_metrics)
    #print(type(run_metrics))

    #print(summary)
    #print(type(summary))
    
    # extract number of lanes and reads
    lane_count = summary.lane_count()
    read_count = summary.size()

    #print(f"Lane count: {summary.lane_count()}")
    #print(f"Read count: {summary.size()}")
    
    # gather data
    data = []

    # collect data for lanes
    for lane in range(lane_count):
        d = {
            "Lane": lane + 1,
            "Yield Gb": np.sum([summary.at(read).at(lane).yield_g() for read in range(read_count)]).round(2),
            "Percent Q30": np.mean([summary.at(read).at(lane).percent_gt_q30() for read in range(read_count)]).round(2),
            "Percent PF": np.mean([summary.at(read).at(lane).percent_pf().mean() for read in range(read_count)]).round(2),
            "Error Rate": np.mean([summary.at(0).at(lane).error_rate().mean(), summary.at(3).at(lane).error_rate().mean()]).round(2)
            }
        data.append(d)
    # combine lanes for total
    d_total = {
        "Lane": "Full Run",
        "Yield Gb": np.sum([lane_d["Yield Gb"] for lane_d in data]),
        "Percent Q30": round(np.mean([lane_d["Percent Q30"] for lane_d in data]), 2),
        "Percent PF": round(np.mean([lane_d["Percent PF"] for lane_d in data]), 2),
        "Error Rate": round(np.mean([lane_d["Error Rate"] for lane_d in data]), 2)
    }
    data.append(d_total)
    # load into dataframe
    df = pd.DataFrame(data)
    # reorder columns
    df = df[['Lane', 'Percent Q30', 'Error Rate', 'Percent PF', 'Yield Gb']]
    #print(df)
    return df


if __name__ == "__main__":
    qc_summary = inter_op_qc("/mnt/dxstream/runs/NovaSeq/241129_A01184_0704_AHJGHTDRX5")
    #with open(output, "w") as f:
    #    qc_summary.to_csv(output, sep="\t", header=True, index=False)
    #print(qc_summary.to_string(index=False))
    #print()
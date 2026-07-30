"""guardian_qc utils_cancer.py

Util functions used by guardian_qc.

Methods
-------

get_qc_summary_file_path()
	gathers all qc_summary_file_paths from the /mnt/dxstream/outputs directory 
    and returns a dataframe with the following columns:
        - seq_run_number
        - worklist
        - qc_file_path
        - sequencer
        - cancer_type

get_run_file_paths()
    gathers all run_summary_file_paths and all sample_sheet_file_paths 
    from the /mnt/dxstream/runs directory and returns a dataframe with
    the following columns:
        - seq_run_number
        - run_qual_filepath
        - sample_sheet_path

helper_get_lane()
    helper function to get the lane information from the sample sheet
    returns a list of lanes present in the sample sheet.

merge_run_and_qc_data()
    merges the run_summary_file_paths and qc_summary_file_paths dataframes
    on the seq_run_number column and returns a merged dataframe with the following columns:
        - seq_run_number
        - run_qual_filepath
        - sample_sheet_path
        - worklist
        - qc_file_path
        - sequencer
        - cancer_type
        - file_status

date:	2026-02-03
author:	Rebecca Sizer

"""
# Import necessary modules
from tools.utils.logger import logging
from tools.utils import config

# Import pandas for data manipulation
import pandas as pd


# This function gets the sample name from the file path 
def get_qc_summary_file_path():
    """
    Gather all file paths to qc_summary files for 
    both the NovaSeqX and NovaSeq6000 platforms.

    params:
        None

    output:
        df
            contains sequence run folder and File paths

    """

    qc_summary_file_paths = []

    qc_patterns = {
        "solid_tumour_v3": config.QC_SUMMARY_PATTERN_ST_V3,
        "solid_tumour": config.QC_SUMMARY_PATTERN_ST,
        "haem_v2": config.QC_SUMMARY_PATTERN_HAEM_V2,
        "haem_v3": config.QC_SUMMARY_PATTERN_HAEM_V3,
    }

    logging.info("Gathering qc_summary file paths from the outputs directory.")

    try:
        
        for run in config.ROOT_FILE_PATH.iterdir():

            if not run.is_dir():
                continue

            if config.RUN_PATTERN_NOVASEQX.match(run.name):
                sequencer = "novaseqx"
            elif config.RUN_PATTERN_NOVASEQ6000.match(run.name):
                sequencer = "novaseq6000"
            else:
                continue  # Skip directories that don't match either pattern

            run_id = run.name

            for worklist_dir in run.iterdir():
                if worklist_dir.is_dir() and config.RUN_PATTERN_SUB.match(worklist_dir.name):
                        
                    worklist = worklist_dir.name

                    for file_path in worklist_dir.iterdir():

                        if file_path.name[0:7] != worklist:
                            continue  # Skip files that don't start with the worklist number

                        for cancer_type, pattern in qc_patterns.items():

                            if pattern.match(file_path.name):

                                qc_summary_file_paths.append(
                                    {
                                        "seq_run_number": run_id,
                                        "worklist": worklist,
                                        "qc_file_path": file_path,
                                        "sequencer": sequencer,
                                        "cancer_type": cancer_type,
                                    }
                                )
                                break

        qc_file_paths = pd.DataFrame(
            qc_summary_file_paths,
            columns=[
                "seq_run_number",
                "worklist",
                "qc_file_path",
                "sequencer",
                "cancer_type"
            ]
        )

        logging.info(f"{len(qc_file_paths)} qc_summary_file_paths loaded into a dataframe.")

        return qc_file_paths

    except FileNotFoundError as e: 
        # Log the error.
        logging.error(f"FileNotFoundError: {e}")



#this function gets the sample name from the file path 
def get_run_file_paths():
    """
    Gather all file paths to run folders 
    
    params: 
        None

    output:
        df
            contains sequence run folder and File paths 
    
    Examples:
        get_run_file_paths()
        
    """
    logging.info("Gathering run file paths from the runs directory.")

    runs = []

    for run in config.NOVASEQX_PATH.iterdir():
        if run.is_dir() and config.RUN_PATTERN_NOVASEQX.match(run.name):
            run_info = {
                "seq_run_number": run.name,
                "run_qual_filepath": str(run),
                "sample_sheet_path": "No sample sheet found",
                "lane": None
            }

            for file in run.iterdir():

                if file.is_file() and config.SAMPLE_SHEET_PATTERN.match(file.name):

                    run_info["sample_sheet_path"] = str(file)

                    lane_info = helper_get_lane(file)
                    run_info["lane"] = lane_info
                    break  # assuming only one sample sheet per run

            runs.append(run_info)
    
    for run in config.NOVASEQ6000_PATH.iterdir():
        
        if run.is_dir() and config.RUN_PATTERN_NOVASEQ6000.match(run.name):
            run_info = {
                "seq_run_number": run.name,
                "run_qual_filepath": str(run),
                "sample_sheet_path": "No sample sheet found",
                "lane" : None
            }

            for file in run.iterdir():
                if file.is_file() and config.SAMPLE_SHEET_PATTERN.match(file.name):
                    run_info["sample_sheet_path"] = str(file)
                    run_info["lane"] = [8] # NovaSeq6000 has 8 lanes which do not need to be separated into individual lanes for this analysis
                    break  # assuming only one sample sheet per run

            runs.append(run_info)

    run_file_paths = pd.DataFrame(
        runs,
        columns=[
            "seq_run_number",
            "run_qual_filepath",
            "sample_sheet_path",
            "lane"
        ]
    )
    logging.info(f"{len(run_file_paths)} run_summary_file_paths loaded into a dataframe.")
    return run_file_paths

############################################################
def helper_get_lane(file_path):

    """
    Gather all file paths to sample sheets 
    
    params: 
        None

    output:
        df
            contains sequence run folder and File paths 
    
    Examples:
        helper_get_lane()
    """

    try:

        df_lane_info = pd.read_csv(file_path, skiprows=15, header=0)

        if "Lane" not in df_lane_info.columns:
            logging.warning(f"Lane column not found in sample sheet: {file_path}")
            return None

        elif df_lane_info["Lane"].isnull().all():
            logging.warning(f"Lane column is empty in sample sheet: {file_path}")
            return None

        else:

            lanes = sorted(df_lane_info["Lane"].unique())

            if lanes == [1]:
                return [1]
            elif lanes == [2]:
                return [2]
            elif lanes == [1, 2]:
                return [1, 2]

    except Exception as e:
        logging.error(f"Error reading sample sheet {file_path}: {e}")
        return None


def merge_run_and_qc_data(df_run_metrics, df_sample_metrics):
    """
    Merge the run_summary_file_paths and qc_summary_file_paths dataframes
    on the seq_run_number column and return a merged dataframe with the following columns:
        - seq_run_number
        - run_qual_filepath
        - sample_sheet_path
        - worklist
        - qc_file_path
        - sequencer
        - cancer_type
        - file_status

        params:
            df_run_metrics: DataFrame containing run_summary_file_paths
            df_sample_metrics: DataFrame containing qc_summary_file_paths
            
        output:
            df_merged: Merged DataFrame containing both run and qc summary file paths
            
        """

    df_merged = df_sample_metrics.merge(df_run_metrics,
                                     on = "seq_run_number",
                                     how = "left")
    

    def check_both_files_present(row):

        run = pd.notna(row["run_qual_filepath"])
        sample = pd.notna(row["qc_file_path"])

        if run and sample:
            return "Yes"
        elif run:
            return "Run QC metrics only"
        elif sample:
            return "Sample QC metrics only"
        else:
            return "No QC data"

    df_merged["file_status"] = df_merged.apply(
        check_both_files_present, 
        axis = 1)
    
    df_merged.to_csv("outputs/file_paths.csv", sep='\t', header=True, index=False)

    counts = (
        df_merged.groupby(["sequencer", "cancer_type"])
        .size()
        .reset_index(name="count")
    )

    logging.info(
        f"Counts of complete and incomplete data sets: "
        f"{df_merged['file_status'].value_counts()}" \
        "\nCounts of data by sequencer and cancer type:" \
        f"{counts}"
    )
    
    return df_merged 

def filter_df(merged_dataframe):
    """
    Filter the merged dataframe to include only rows where:
    - file_status is 'Yes'
    - lane is not None
    
    params:
        merged_dataframe: DataFrame containing merged run and qc summary file paths
    
    output:
        filtered_df: Filtered DataFrame containing only rows that meet the criteria
    """

    pass_filter = []

    for _, row in merged_dataframe.iterrows():

        if row["file_status"] == "Yes" and row["lane"] is not None:
            pass_filter.append(row)

    filtered_df = pd.DataFrame(pass_filter)
    logging.info(f"Filtered dataframe contains {len(filtered_df)} rows after applying QC and lane filters.")
    return filtered_df


#test functions in script
if __name__ == "__main__":
    output_run_folder = get_run_file_paths()
    output_qc_file = get_qc_summary_file_path()
    merged_output = merge_run_and_qc_data(output_run_folder, output_qc_file)
    filtered_output = filter_df(merged_output)

# exploration of merged_output dataframe
# 827 runs do not have QC_Summary files
# 46 runs do not have SampleSheet files
# 627 rows pass filtering criteria (file_status == 'Yes' and lane is not None)
    #print(filtered_output.isna().sum())
    #print(f"Missing lane values: {filtered_output['lane'].isna().sum()}")
    #print(f"Missing sample_qc_path values: {filtered_output['qc_file_path'].isna().sum()}")
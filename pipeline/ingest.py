"""guardian_qc ingest.py

Functions used to ingest the data used by guardian_qc.

This script pulls run, sample qc and sample sheet file paths into
a single dataframe. It then filters the dataframe to remove data
with missing values, and then uses the interop package to remove
any runs that have failed run qc. The script then pulls all sample
qc_metrics from all samples on runs that have passed these filters. 

This script pulls data for solid tumour (ST) and Haem DNA runs from
both the NovaSeq6000 and NovaSeqX.

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

filter_df()
    filters the merged dataframe to include only rows where:
    - file_status is 'Yes'
    - lane is not None  

filter_run_qc()
    filters the dataframe to include only runs that pass run level QC metrics.

date:	2026-02-03
author:	Rebecca Sizer

"""
# Import necessary modules
from utils.logger import logging
from qc.inter_op import inter_op_qc
import config
from pathlib import Path

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

            elif config.RUN_PATTERN_NOVASEQX.match(run.name):
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

                        if not file_path.name.startswith(worklist):
                            continue
                            # Skip files that don't start with the worklist number

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
        return pd.DataFrame()



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

                lanes = sorted(df_lane_info["Lane"].dropna().astype(int).unique().tolist())
                return lanes

        except Exception as e:
            logging.error(f"Error reading sample sheet {file_path}: {e}")
            return None

    logging.info("Gathering run file paths from the runs directory.")

    runs = []

    try:
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

    except FileNotFoundError as e: 
        # Log the error.
        logging.error(f"FileNotFoundError: {e}")
        return pd.DataFrame()


def get_run_sample_file_paths(df_run_metrics, df_sample_metrics):
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

    df_merged = df_run_metrics.merge(df_sample_metrics,
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
    
    df_merged.to_csv(config.QC_FILE_PATHS, sep='\t', header=True, index=False)

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

        lane = row["lane"]

        # Check that lane is present
        if lane is None:
            lane_present = False
        elif isinstance(lane, float) and pd.isna(lane):
            lane_present = False
        elif isinstance(lane, list):
            lane_present = len(lane) > 0
        else:
            lane_present = True

        if row["file_status"] == "Yes" and lane_present:
            pass_filter.append(row)

        else:
            continue

    filtered_df = pd.DataFrame(pass_filter)
    logging.info(f"Filtered dataframe contains {len(filtered_df)} rows after applying QC and lane filters.")

    return filtered_df

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

    logging.info(df_pass["pass_run_qc"].value_counts())

    return df_pass 


def sample_level_qc(df):
    """
    Function to gather sample-level QC metrics for each sample in the dataframe.
    
    params:
        df: pd.DataFrame containing the merged run and qc summary file paths.
    
    output:
        df: pd.DataFrame containing the summarized sample-level QC metrics for each sample.
    """

    # Helper function to extract values from the QC summary file
    def extract_values_from_qc_summary(file_path):
        """
        Function to extract values from the QC summary file.
        
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
                    "picard_gc_dropout": row["picard_gc_dropout"],
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
    
    summary_qc_metrics = []

    for _, row in df.iterrows():

        if row["pass_run_qc"] == "Yes":

            summary_qc_file_path = row["qc_file_path"]
            sample_qc = extract_values_from_qc_summary(summary_qc_file_path)

            for sample in sample_qc:

                sample["sequencer"] = row["sequencer"]
                sample["cancer_type"] = row["cancer_type"]
                sample['worklist'] = row['worklist']
                summary_qc_metrics.append(sample)

    summary_qc_metrics_df = pd.DataFrame(summary_qc_metrics)

    return summary_qc_metrics_df


#test functions in script
if __name__ == "__main__":
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
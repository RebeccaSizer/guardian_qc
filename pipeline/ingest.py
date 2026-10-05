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


Graph functions
----------------
pie_chart_run_metric_pass_rate()
    plots a pie chart showing the proportion of runs passing/failing qc, and the 
    reasons why

bar_chart_sample_count_by_sequencer_and_cancer_type()
    plots a bar chart showning the number of samples per sequencing, cancer type,
    and capture version. 

comparing_sequencer_cancer_type()
    plots a PCA graph for each metric split across sequencer, and cancer_type + 
    capture version. 

test_group_differences()
    calculates the statistical difference between groups (sequencer, cancer_type 
    + version)


Overall ingestion function
--------------------------
ingest_data()
    Pulls in file paths, filters by run qc metrics, extract sample qc
    metrics and place into a dataframe. 

date:	2026-02-03
author:	Rebecca Sizer

"""

##############################
# Import necessary modules
##############################

from utils.logger import logging
from qc.inter_op import inter_op_qc
import config
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
from scipy.stats import kruskal
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from scipy import stats

#################################
# Set script variables
#################################

metric_cols = [
        "bcftools_ts",
        "bcftools_tv",
        "bcftools_tstv",
        "bcftools_variants",
        "bcftools_snvs",
        "bcftools_indels",
        "picard_mode_insert",
        "picard_mean_insert",
        "picard_median_insert",
        "picard_mad_insert",
        "picard_total_reads",
        "picard_pf_reads",
        "picard_pf_q30_bases",
        "picard_read_length",
        "picard_at_dropout",
        "picard_gc_dropout",
        "picard_fold_enrichment",
        #"picard_fold80",
        "picard_mean_target_coverage",
        "picard_median_target_coverage",
        "picard_target_bases_20x",
        "picard_target_bases_30x",
        "picard_target_bases_50x",
        "picard_target_bases_100x",
        "fastqc_duplication_rate",
        #"fastqc_basic_status", This metric is str
        "fastp_duplication_rate",
    ]

#####################################
# Inget data functions 
#####################################

# This function gets the sample name from the file path 
def get_qc_summary_file_path():
    """
    Gather all file paths to sample qc_summary files for 
    both the NovaSeqX and NovaSeq6000 platforms for Heam and ST.

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

            elif config.RUN_PATTERN_NOVASEQX.match(run.name): # Get name attribute of the run path (the final folder)
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



# This function gets the sample name from the file path 
def get_run_file_paths():
    """
    Gather all file paths to run folders 
    
    params: 
        None

    output:
        df
            contains:
                - Sequence run number
                - Run qulaity file path 
                - Sample sheet file path
                - Lane
    
    Examples:
        get_run_file_paths()
        
    """
    def helper_get_lane(file_path):

        """
        Gather all file paths to sample sheets 
        
        params: 
            None

        output:
            Returns Integer List of lane values, or None if not available  
        
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

                        lane_info = helper_get_lane(file) # Need Lane information for NovaSeqX as can be 1,2 or both
                        run_info["lane"] = lane_info
                        break  # assuming only one sample sheet per run

                runs.append(run_info)
        
        for run in config.NOVASEQ6000_PATH.iterdir():
            
            if run.is_dir() and config.RUN_PATTERN_NOVASEQ6000.match(run.name):
                run_info = {
                    "seq_run_number": run.name,
                    "run_qual_filepath": str(run),
                    "sample_sheet_path": "No sample sheet found",
                    "lane" : None # Lane information not needed for NovaSeq6000 as only 8 lane option. 
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


# Merge the two data frames (All details in one place)
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
            df_run_metrics: DataFrame containing run_summary_file_paths, 
            df_sample_metrics: DataFrame containing qc_summary_file_paths
            
        output:
            df_merged: Merged DataFrame containing both run and qc summary file paths
            
        """

    
    def check_both_files_present(row):
        """
        This function adds an additional column to indicate if all necassary files are present
        
        params:
            dataframe with columns run_qual_filepath and qc_file_path
            
        output:
            string Yes, Run QC metrics only, Sample QC metrics only, or No QC data"""

        try:
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

        except KeyError as e:
            logging.error(f"Missing required column {e}")

    try:
        df_merged = df_run_metrics.merge(df_sample_metrics,
                                        on = "seq_run_number",
                                        how = "left")

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

    except KeyError as e:
        logging.error(f"Merge failed due to missing key: {e}")


# Filter the dataframe to only include samples that have both run, sample paths and lane information
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

    try:

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

    except KeyError as e:
        logging.error(f"Missing columns: {e}")

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

    try:

        for _, row in df.drop_duplicates("seq_run_number").iterrows():

            run_folder_path = row["run_qual_filepath"]
            run_folder_name = Path(run_folder_path).name

            run_df = inter_op_qc(run_folder_path)
            print(run_df)

            # Get Lane information to select correct values from the run_qc pulled using interop
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

            q30 = pd.to_numeric(run_columns["Percent Q30"], errors="coerce")
            error_rate = pd.to_numeric(run_columns["Error Rate"], errors="coerce")

            if q30 >= 80 and (error_rate <= 2 or pd.isna(error_rate)):
                logging.info(f'q30:{q30}, error rate:{error_rate}')
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

        # Return data frame of all with pass_run_qc column present
        return df_pass 

    except KeyError as e:
        logging.error(f"Missing column: {e}")


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

    try:
        for _, row in df.iterrows():

            if row["pass_run_qc"] == "Yes": # Only pull for samples that pass run QC 

                summary_qc_file_path = row["qc_file_path"]
                sample_qc = extract_values_from_qc_summary(summary_qc_file_path)

                for sample in sample_qc:

                    sample["sequencer"] = row["sequencer"]
                    sample["cancer_type"] = row["cancer_type"]
                    sample['worklist'] = row['worklist']
                    sample['assay_type'] = row['sequencer'] + '_' + row['cancer_type']
                    summary_qc_metrics.append(sample)

        summary_qc_metrics_df = pd.DataFrame(summary_qc_metrics)

        return summary_qc_metrics_df

    except KeyError as e:
        logging.error(f"Missing column: {e}")

#################################################
# plot graphs for ingestion
#################################################

def pie_chart_run_metric_pass_rate(df):
    counts = df['pass_run_qc'].value_counts()

    labels = [
        "Pass" if label == "Yes" else label
        for label in counts.index
    ]

    colours = {
        'Yes': '#4ECFF7',  # Blue
        'Percent Q30 < 80 AND Error rate > 2': '#AF4C82',  # Pink
        'Error rate > 2': '#FFC107'  # Yellow
    }

    pie_colours = [colours.get(label, '#D3D3D3') for label in counts.index]

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 6))

    wedges, _, autotexts = ax.pie(
        counts,
        labels=None,
        colors=pie_colours,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12},
        pctdistance=1.1,
    )

    # Equal aspect ratio keeps the pie circular
    ax.axis("equal")

    # Title
    ax.set_title(
        "Proportion of Runs Passing Run-Level QC",
        fontsize=14,
        fontweight="bold",
    )

    # Legend
    ax.legend(
        wedges,
        labels,
        title="Run QC Status",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
    )

    plt.tight_layout()

    # Save figure
    plt.show()
    plt.savefig('outputs/graphs/data_exploration/pie_chart_pass_run_qc.png')


def bar_chart_sample_count_by_sequencer_and_cancer_type(df):

    counts = df[['sequencer', 'cancer_type']].value_counts().reset_index(name="count")

    # Create a combined label for the x-axis
    counts["label"] = counts["sequencer"] + ": " + counts["cancer_type"]

    fig, ax = plt.subplots(figsize=(8, 6))

    plt.figure(figsize=(8, 6))
    plt.bar(counts["label"], counts["count"])

    plt.xlabel("Sequencer / Cancer Type")
    plt.ylabel("Number of Samples")
    plt.title("Samples by Sequencer and Cancer Type")
    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()
    plt.savefig('outputs/graphs/data_exploration/sample_count_by_cancer_type_and_sequencer.png')


def comparing_sequencer_cancer_type(df):

    # Scale
    df_copy = df.copy()
    X = df_copy[metric_cols].fillna(df_copy[metric_cols].median())
    X_scaled = RobustScaler().fit_transform(X)

    # PCA
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    df_copy["PC1"] = coords[:, 0]
    df_copy["PC2"] = coords[:, 1]

    loadings = pd.DataFrame(
        pca.components_.T,
        index=metric_cols,
        columns=["PC1", "PC2"]
    )
    print(loadings["PC1"].abs().sort_values(ascending=False).head(10))
    print(loadings["PC2"].abs().sort_values(ascending=False).head(10))

    print(pca.explained_variance_ratio_)

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # By sequencer
    for seq in df_copy["sequencer"].unique():
        mask = df_copy["sequencer"] == seq
        axes[0].scatter(df_copy.loc[mask,"PC1"], df_copy.loc[mask,"PC2"],
                        label=seq, alpha=0.3, s=6)

    axes[0].set_title("By sequencer")
    axes[0].set_xlabel(f"PC1 ({var1:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({var2:.1f}%)")
    axes[0].legend(markerscale=4, fontsize=8)

    # By cancer type
    for ct in df_copy["cancer_type"].unique():
        mask = df_copy["cancer_type"] == ct
        axes[1].scatter(df_copy.loc[mask,"PC1"], df_copy.loc[mask,"PC2"],
                        label=ct, alpha=0.3, s=6)
    axes[1].set_title("By cancer type")
    axes[1].set_xlabel(f"PC1 ({var1:.1f}%)")
    axes[1].legend(markerscale=4, fontsize=8)

    # By both — combine labels
    
    df_copy["group"] = df_copy["sequencer"] + " | " + df_copy["cancer_type"]
    for grp in df_copy["group"].unique():
        mask = df_copy["group"] == grp
        axes[2].scatter(df_copy.loc[mask,"PC1"], df_copy.loc[mask,"PC2"],
                        label=grp, alpha=0.3, s=6)
    axes[2].set_title("By sequencer + cancer type")
    axes[2].set_xlabel(f"PC1 ({var1:.1f}%)")
    axes[2].legend(markerscale=4, fontsize=8, 
                bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig("outputs/graphs/data_exploration/pca_by_group.png", dpi=150, bbox_inches="tight")


def test_group_differences(df, cols, group_col):
    """
    For each metric, run Kruskal-Wallis across groups.

    Returns a summary DataFrame sorted by p-value,
    including the number of observations in each group.
    """

    results = []

    for metric in metric_cols:
        groups = [
            group[metric].dropna()
            for _, group in df.groupby(group_col)
        ]

        if all(len(group) > 1 for group in groups):
            stat, p_value = kruskal(*groups)

            results.append({
                "metric": metric,
                "statistic": stat,
                "p_value": p_value
            })

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    # Benjamini-Hochberg FDR correction
    _, p_values_corrected, _, _ = multipletests(
        results_df["p_value"],
        alpha=0.05,
        method="fdr_bh"
    )

    results_df["p_value_corrected"] = p_values_corrected

    results_df["significant"] = (
        results_df["p_value_corrected"] < 0.05
    )

    return results_df.sort_values("p_value_corrected")


###################################################
# Ingest data using above functions
def ingest_data():
    """
    This function produces a dataframe containing all of the sample level quality metrics
    pulled from any run which passed run level qc. Metrics pulled through include:

    - bcftools metrics
        bcftools_ts — Number of transitions (e.g. A↔G or C↔T) identified in the variant calls.
        bcftools_tv — Number of transversions (e.g. A↔C, A↔T, C↔G, etc.) identified in the variant calls.
        bcftools_tstv — Transition-to-transversion (Ti/Tv) ratio. This can provide an indication of variant call quality and composition.
        bcftools_variants — Total number of variants identified in the sample.
        bcftools_snvs — Number of single-nucleotide variants (SNVs) identified.
        bcftools_indels — Number of insertions and deletions (indels) identified.
    - Picards metrics
        picard_mode_insert — Most common insert size observed in the sequencing reads.
        picard_mean_insert — Mean insert size across the sequencing reads.
        picard_median_insert — Median insert size across the sequencing reads.
        picard_mad_insert — Median absolute deviation (MAD) of insert sizes; indicates how much insert sizes vary around the median.
        picard_total_reads — Total number of sequencing reads examined.
        picard_pf_reads — Number of passing-filter (PF) reads, i.e. reads that pass the sequencing platform's quality filter.
        picard_pf_q30_bases — Proportion/number of PF bases with a Phred quality score ≥30, indicating high base-call quality.
        picard_read_length — Length of the sequencing reads in base pairs.
        picard_at_dropout — AT dropout, measuring uneven coverage associated with AT-rich regions.
        picard_gc_dropout — GC dropout, measuring uneven coverage associated with GC-rich regions.
        picard_fold_enrichment — Measures how much sequencing coverage is enriched in the target regions compared with the expected/background coverage.
        picard_fold80 — Fold 80 base penalty; indicates how much additional sequencing would theoretically be required to achieve uniform coverage. Lower values generally indicate more uniform coverage.
        picard_mean_target_coverage — Average sequencing depth/coverage across the targeted regions.
        picard_median_target_coverage — Median sequencing depth/coverage across the targeted regions.
        picard_target_bases_20x — Percentage/proportion of target bases covered by at least 20×.
        picard_target_bases_30x — Percentage/proportion of target bases covered by at least 30×.
        picard_target_bases_50x — Percentage/proportion of target bases covered by at least 50×.
        picard_target_bases_100x — Percentage/proportion of target bases covered by at least 100×.
    - FastQC metrics
        fastqc_duplication_rate — Proportion of sequencing reads that are duplicates. High duplication can indicate PCR amplification or low library complexity.
        fastqc_basic_status — Overall FastQC basic quality status, indicating whether the basic FastQC checks passed or identified potential warnings/failures.
   - Fastp metrics
        fastp_duplication_rate — Proportion of reads identified as duplicates by Fastp, providing another measure of library complexity/duplication.
    
    params:
        None
    
    output:
        Dataframe containing sample level qc metrics
        CSV file of all file paths and samples used
        CSV of all sample level QC metrics
    """
    logging.info(f"Starting ingestion pipeline...")
    df_sample = get_qc_summary_file_path()
    df_run = get_run_file_paths()

    df = get_run_sample_file_paths(df_run, df_sample)
    df.to_csv(config.QC_FILE_PATHS, sep="\t", index=False)

    # Filter the data
    df = filter_df(df)

    df = filter_run_qc(df)
    pie_chart_run_metric_pass_rate(df)
    
    bar_chart_sample_count_by_sequencer_and_cancer_type(df)

    df.to_csv(config.FILTERED_RUN_QC_DATA, sep="\t", index=False)

    # Extract summary QC metrics
    summary_qc_metrics_df = sample_level_qc(df)
    comparing_sequencer_cancer_type(summary_qc_metrics_df)

    summary_qc_metrics_df.to_csv(config.SUMMARY_QC_METRICS, sep = "\t", index = False)

    logging.info(
            f"Summary QC metrics written to "
            f"{config.SUMMARY_QC_METRICS} "
            f"({len(summary_qc_metrics_df)} rows)"
        )

    return summary_qc_metrics_df


if __name__ == "__main__":

    # Ingest the data 
    df = ingest_data()

    # Create a copy to create graphs
    df_metrics = df.copy()

    # split data by cancer type for stats
    df_st = df_metrics[
        (df_metrics['cancer_type'] == 'solid_tumour') |
        (df_metrics['cancer_type'] == 'solid_tumour_v3')
    ].copy()

    df_haem = df_metrics[
        (df_metrics['cancer_type'] == 'haem_v2') |
        (df_metrics['cancer_type'] == 'haem_v3')
    ].copy()


    # Complete stats tests
    results_seq_st = test_group_differences(
        df_st, metric_cols, "sequencer"
    )

    results_seq_haem = test_group_differences(
        df_haem, metric_cols, "sequencer"
    )

    results_cancer = test_group_differences(
        df_metrics, metric_cols, "cancer_type"
    )

    results_both_st = test_group_differences(
        df_st, metric_cols, "assay_type"
    )

    results_both_haem = test_group_differences(
        df_haem, metric_cols, "assay_type"
    )

    print("=== By sequencer (ST) ===")
    print(
        results_seq_st[
            ["metric", "p_value_corrected", "p_value", "significant"]
        ].to_string()
    )

    print("\n=== By sequencer (Haem) ===")
    print(
        results_seq_haem[
            ["metric", "p_value_corrected", "p_value", "significant"]
        ].to_string()
    )

    print("\n=== By cancer type ===")
    print(
        results_cancer[
            ["metric", "p_value_corrected", "p_value", "significant"]
        ].to_string()
    )

    print("\n=== By assay type (ST) ===")
    print(
        results_both_st[
            ["metric", "p_value_corrected", "p_value", "significant"]
        ].to_string()
    )

    print("\n=== By assay type (Haem) ===")
    print(
        results_both_haem[
            ["metric", "p_value_corrected", "p_value", "significant"]
        ].to_string()
    )
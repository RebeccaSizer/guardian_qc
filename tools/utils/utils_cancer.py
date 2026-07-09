"""guardian_qc utils

Util functions used by guardian_qc.

Methods
-------

get_path_info()
	given a file path, this function extracts the files directory, the 
	base file name, and its ending
get_help_text()
	returns a formatted string describing the seqQscorer help text
clf_full_names(abbr)
	given one of the abbreviations, this function returns the full name of
	the algorithm. Used to clarify the terminal output
get_best_classifier(utils_dir, species, assay, run_type, feature_sets, fs_suffix, metric)
	given the user specifications from seqQscorer, this function parses a text table
	in order to return the classifier and feature selection specifications that are most 
	recommendable for the application
read_in_measure_table(utils_dir, species, assay, run_type, feature_sets, fs_suffix, metric)
	seqQscorer prints a table with machine learning evaluation measures for different decision 
	thresholds. The source file with this information is parsed by this function
def get_clf_algos()
	this function creates and returns a dictionary of default classifier configuratons

date:	2026-02-03
author:	Rebecca Sizer

"""

#Check all imports are used

#from sklearn.ensemble import RandomForestClassifier
#from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import SVC
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.naive_bayes import GaussianNB
#from sklearn.neural_network import MLPClassifier
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.ensemble import AdaBoostClassifier
#from sklearn.tree import ExtraTreeClassifier

#import os
#import json
#import pandas as pd
#import subprocess
#from terminaltables import AsciiTable
from pathlib import Path
from tools.utils.logger import logging
import pandas as pd
import re

#this function gets the sample name from the file path 
def get_qc_summary_file_path():
    """
    Extract a sample name from a sequencing file path.
    
    params: 
        None

    output:
        str
            The base filename with path and extensions removed.
    
    Examples:
        get_file_name("/data/sample.fastq.gz")
        'sample'
        getFileName("reads/sample.fq")
        'sample'
    """
    #Path allows you to manipulate windows paths on Unix machines
    root_file_path = Path("/mnt/dxstream/outputs")
    qc_summary_file_paths = []

    run_pattern_novaseqx = re.compile(r"^\d{6,8}_LH00537_\d{4}_[A-Z0-9]{10}$")
    run_pattern_novaseq6000 = re.compile(r"^\d{6,8}_A01184_\d{4}_[A-Z0-9]{10}$")
    
    run_pattern_sub = re.compile(r"^[0-9]{7}$")
    
    qc_summary_pattern_st_v3 = re.compile(r"^[0-9]{7}\.RMH200STv3\.qc_summary\.tsv$")
    qc_summary_pattern_st = re.compile(r"^[0-9]{7}\.RMH200ST\.qc_summary\.tsv$")
    qc_summary_pattern_haem_v2 = re.compile(r"^[0-9]{7}\.RMHhaemV2\.qc_summary\.tsv$")
    qc_summary_pattern_haem_v3 = re.compile(r"^[0-9]{7}\.RMHhaemV3\.qc_summary\.tsv$")
    

    try:
        
        for run in root_file_path.iterdir():

            if run.is_dir() and run_pattern_novaseqx.match(run.name):
            
                run_id = run.name
                build_path = run
            

                for i in build_path.iterdir():
                    if i.is_dir() and run_pattern_sub.match(i.name):
                        
                        build_path = i

                        for file_path in build_path.iterdir():
                            if qc_summary_pattern_st_v3.match(file_path.name):
                        
                                build_path = file_path
                                qc_summary_file_paths.append({
                                    "seq_run_number" : run_id,
                                    "qc_file_path" : build_path,
                                    "sequencer" : "novaseqx",
                                    "cancer_type" : "solid_tumour_v3"
                                })
                            
                            elif qc_summary_pattern_st.match(file_path.name):
                        
                                build_path = file_path
                                qc_summary_file_paths.append({
                                    "seq_run_number" : run_id,
                                    "qc_file_path" : build_path,
                                    "sequencer" : "novaseqx",
                                    "cancer_type" : "solid_tumour"
                                })
                            
                            elif qc_summary_pattern_haem_v2.match(file_path.name):
                        
                                build_path = file_path
                                qc_summary_file_paths.append({
                                    "seq_run_number" : run_id,
                                    "qc_file_path" : build_path,
                                    "sequencer" : "novaseqx",
                                    "cancer_type" : "haem_v2"
                                })
                            
                            elif qc_summary_pattern_haem_v3.match(file_path.name):
                        
                                build_path = file_path
                                qc_summary_file_paths.append({
                                    "seq_run_number" : run_id,
                                    "qc_file_path" : build_path,
                                    "sequencer" : "novaseqx",
                                    "cancer_type" : "haem_v3"
                                })

                            else:
                                continue
                    else:
                        continue
            
            elif run.is_dir() and run_pattern_novaseq6000.match(run.name):
            
                run_id = run.name
                build_path = run
            

                for i in build_path.iterdir():
                    if i.is_dir() and run_pattern_sub.match(i.name):
                        
                        build_path = i

                        for file_path in build_path.iterdir():
                            if qc_summary_pattern_st_v3.match(file_path.name):
                        
                                build_path = file_path
                                qc_summary_file_paths.append({
                                    "seq_run_number" : run_id,
                                    "qc_file_path" : build_path,
                                    "sequencer" : "novaseq6000",
                                    "cancer_type" : "solid_tumour_v3"
                                })
                            
                            elif qc_summary_pattern_st.match(file_path.name):
                        
                                build_path = file_path
                                qc_summary_file_paths.append({
                                    "seq_run_number" : run_id,
                                    "qc_file_path" : build_path,
                                    "sequencer" : "novaseq6000",
                                    "cancer_type" : "solid_tumour"
                                })
                            
                            elif qc_summary_pattern_haem_v2.match(file_path.name):
                        
                                build_path = file_path
                                qc_summary_file_paths.append({
                                    "seq_run_number" : run_id,
                                    "qc_file_path" : build_path,
                                    "sequencer" : "novaseq6000",
                                    "cancer_type" : "haem_v2"
                                })
                            
                            elif qc_summary_pattern_haem_v3.match(file_path.name):
                        
                                build_path = file_path
                                qc_summary_file_paths.append({
                                    "seq_run_number" : run_id,
                                    "qc_file_path" : build_path,
                                    "sequencer" : "novaseq6000",
                                    "cancer_type" : "haem_v3"
                                })

                            else:
                                continue
                    else:
                        continue 
            else:
                continue
        
        qc_file_paths = pd.DataFrame(
            qc_summary_file_paths,
            columns=[
                "seq_run_number",
                "qc_file_path",
                "sequencer",
                "cancer_type"
            ]
        )

        logging.info(f"{len(qc_file_paths)} qc_summary_file_paths loaded into a dataframe.")
        return qc_file_paths           
                
    except FileNotFoundError as e: 
        # Log the error.
        logging.error(f"Variant Parser Error: Uploaded variant file '{filename}' not found: {e}")

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
    runs = []
    novaseqx_path = Path("/mnt/dxstream/runs/NovaSeqX")
    novaseq6000_path = Path("/mnt/dxstream/runs/NovaSeq")

    run_pattern_novaseqx = re.compile(r"^\d{6,8}_LH00537_\d{4}_[A-Z0-9]{10}$")
    run_pattern_novaseq6000 = re.compile(r"^\d{6,8}_A01184_\d{4}_[A-Z0-9]{10}$")

    for run in novaseqx_path.iterdir():
        if run.is_dir() and run_pattern_novaseqx.match(run.name):
            runs.append({
                "seq_run_number": run.name,
                "run_qual_filepath": str(run),
            })
    
    for run in novaseq6000_path.iterdir():
        
        if run.is_dir() and run_pattern_novaseq6000.match(run.name):
            runs.append({
                "seq_run_number": run.name,
                "run_qual_filepath": str(run),
            })

    run_file_paths = pd.DataFrame(
        runs,
        columns=[
            "seq_run_number",
            "run_qual_filepath",
        ]
    )
    logging.info(f"{len(run_file_paths)} run_summary_file_paths loaded into a dataframe.")
    return run_file_paths


def merge_run_and_qc_data(df_run_metrics, df_sample_metrics):

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

    logging.info(
        f"Counts of complete and incomplete data sets: "
        f"{df_merged['file_status'].value_counts()}"
        "\nCounts of data by sequencer and cancer type:" \
        f"{df_merged.value_counts(['sequencer', 'cancer_type'])}"
    )
    
    return df_merged 

#test functions in script
if __name__ == "__main__":
    output_run_folder = get_run_file_paths()
    output_qc_file = get_qc_summary_file_path()
    merged_output = merge_run_and_qc_data(output_run_folder, output_qc_file)
    print(merged_output)
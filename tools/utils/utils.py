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
from tools.utils.logger import logger

#this function gets the sample name from the file path 
def get_file_name(file_path):
    """
    Extract a sample name from a sequencing file path.

    This function removes directory paths and common sequencing
    file extensions (.tsv) from the input file path,
    returning a clean base filename suitable for use as a sample ID.
    
    params: 
        file_path : str
            Full or relative path to a sequencing file.

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
    p = Path(file_path)  #Path navigates to that file path, newer and faster then os
    name = p.stem 
    suffix = p.suffix

    try:
        if suffix == ".tsv":
            name = Path(name).stem #.stem gives the final component of the path without the suffix
            print(f"Path(name).stem = {name}")

        else:
            print(f"incorrect file type found: {name}")
            logger.error(f"incorrect file type: {name}")
    
    except FileNotFoundError as e: 
        # Log the error.
        logger.error(f"Variant Parser Error: Uploaded variant file '{filename}' not found: {e}")

    return name


#test functions in script
if __name__ == "__main__":
    file = "/C:/Users/nb28589/Desktop/project_test/2602781.RMH200ST.qc_summary.tv"
    output = get_file_name(file)
    print(output)


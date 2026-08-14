import pandas as pd
import config
import numpy as np
from utils.logger import logging
from sklearn import preprocessing

"""
guardian_qc preprocessing.py

Preprocessing and feature engineering utilities for Guardian-QC.

This module prepares QC summary data for downstream exploratory analysis,
dimensionality reduction and unsupervised machine-learning models such as
Isolation Forest.

The preprocessing workflow includes:

1. Data exploration
    - Inspect dataset dimensions and column types
    - Summarise missing values
    - Identify duplicate rows
    - Examine distributions and summary statistics
    - Assess feature variance and correlations
    - Explore data by assay, sequencer and other relevant batch variables

2. Data cleaning and preprocessing
    - Handle missing values
    - Remove duplicate rows
    - Correct data types
    - Remove irrelevant or non-informative columns
    - Encode categorical variables
    - Remove low-variance features
    - Scale/normalise numerical features

3. Feature engineering
    - Create error-rate-derived features
    - Combine SNV and indel counts into total variant counts
    - Create variant ratios such as Ti/Tv
    - Extract useful information such as sequencer from run identifiers
    - Create coverage-derived features such as coverage pass rate
    - Apply transformations such as log transformation to highly skewed QC metrics

4. Batch-effect assessment and correction
    - Identify potential batch effects associated with sequencing runs,
      sequencers, assays or other technical variables
    - Calculate batch-aware statistics such as z-scores where appropriate
    - Assess whether batch correction is required before downstream modelling

5. Final feature preparation
    - Confirm that the final feature matrix contains valid numerical values
    - Check for remaining missing or infinite values
    - Verify feature distributions and correlations
    - Return a model-ready feature matrix

The intended workflow is:

    Raw QC data
        |
        v
    Data exploration
        |
        v
    Data cleaning
        |
        v
    Feature engineering
        |
        v
    Missing-value handling
        |
        v
    Feature selection
        |
        v
    Batch-effect assessment/correction
        |
        v
    Feature scaling
        |
        v
    Final model-ready data


Functions
---------
load_data()
    Load the qc_summary csv file into a dataframe

explore_qc_data()
    Perform initial exploration and summarisation of the QC dataset.

remove_duplicates()
    Identify and remove duplicate observations.

correct_data_types()
    Convert QC metrics to appropriate numerical or categorical data types.

remove_irrelevant_features()
    Remove columns that should not be used as model features, such as
    identifiers, file paths or other metadata.

engineer_qc_features()
    Create derived QC features from existing metrics.

handle_missing_values()
    Handle missing values using the selected imputation or filtering strategy.

remove_low_variance_features()
    Remove features with little or no variation across samples.

encode_categorical_features()
    Encode categorical QC variables for downstream analysis.

assess_feature_distributions()
    Identify highly skewed features and determine whether transformations
    are appropriate.

transform_skewed_features()
    Apply transformations such as log transformation to skewed numerical
    features.

assess_feature_correlations()
    Calculate feature correlations and identify highly correlated features.

assess_batch_effects()
    Investigate variation associated with sequencing runs, sequencers,
    assays or other batch variables.

calculate_batch_z_scores()
    Calculate batch-aware z-scores for QC features where appropriate.

scale_features()
    Scale numerical features prior to PCA or machine-learning analysis.

preprocess_qc_data()
    Run the complete preprocessing and feature-engineering workflow and
    return a model-ready feature matrix.

Date: 2026-08-14
Author: Rebecca Sizer
"""

# Load and audit the raw qc summary data
def load_data(file_path):

    summary_qc_df = pd.read_csv(file_path, header=0, sep="\t")

    return summary_qc_df

def explore_qc_data(data_frame):
    """
    This function explores the data and reports:
        - Number ofd Rows
        - Number of Columns
        - Number of samples split by sequencer and cancer type
        - Number of duplicated rows
        - Number of missing values
    
    params:
        dataframe: containing qc summary metrics
    
    output: None
    """

    logging.info("=================================== ")
    logging.info("Summary of the data: ")
    logging.info("=================================== ")
    logging.info("\n")
    logging.info(f"Number of Rows: {data_frame.shape[0]}")
    logging.info(f"Number of Columns: {data_frame.shape[1]}")
    logging.info("------------------------------------")
    logging.info(f"QC metrics in table:\n {data_frame.columns.tolist()}")
    logging.info("------------------------------------")
    logging.info(f"Split by sequencer and cancer type: {data_frame[['sequencer','cancer_type']].value_counts()}")

    logging.info("=================================== ")
    logging.info("Quality of the data: ")
    logging.info("=================================== ")
    logging.info("\n")
    logging.info(f"Duplicte rows: {data_frame.duplicated().sum()} / {data_frame.shape[0]}")
    logging.info("------------------------------------")
    logging.info(f"Info about the data:")
    data_frame.info()
    #logging.info(f"Describe the data:\n {data_frame.describe()}")
    logging.info("------------------------------------")
    logging.info(f"Missing values by columns: "
                 f"{data_frame.isna().sum()}")
    logging.info("------------------------------------")
    logging.info(f"Number of zero values in each column: {(data_frame == 0).sum()}")

def remove_duplicates(data_frame):
    """
    This function removes rows where the whole row is 
    duplicated elsewhere in the dataframe
    
    params:
        dataframe: containing qc summary metrics
        
    output
        dataframe: with duplicate rows removed 
    """

    logging.info(f"Removing any duplicate rows...") # Should I remove duplicate sample names - will this bias the model
    original_len = len(data_frame)
    deduplicated_df = data_frame.drop_duplicates()
    final_len = len(deduplicated_df)
    logging.info(f"Total rows removed due to duplication: {original_len - final_len}")

    return deduplicated_df

def correct_data_types(data_frame):
    """
    This function expands columns where qc metric values are of type string
    and expands them to numeric values using LabelEncoder. 
    This includes coluns for:
        - Picards fold_80
        - fast_qc_basic_status
    
    This does not include information for sample_name, Sequencer or cancer_type
    as this information will be used to split the data into sub groups rather 
    than been used to train the model """

    str_columns = ["picard_fold80", "fastqc_basic_status"]

    le = preprocessing.LabelEncoder()
    data_frame_columns = data_frame.copy()

    logging.info(f"Starting encoding of str values in the dataframe...")
    for column in str_columns:
        logging.info(f"Number of unique values in {column}: {data_frame[column].value_counts()}")
        data_frame[f"{column}_encoded"] = le.fit_transform(data_frame[column])

    logging.info(f"Encoding complete.\n"
                 f"Previous number of columns: {data_frame_columns.shape[1]}\n"
                 f"New value of columns: {data_frame.shape[1]}")

    return data_frame


if __name__ == "__main__":
    qc_summary_df = load_data(config.SUMMARY_QC_METRICS)
    explore_qc_data(qc_summary_df)
    deduplicated_df = remove_duplicates(qc_summary_df)
    encoded_df = correct_data_types(deduplicated_df)
    print(encoded_df)

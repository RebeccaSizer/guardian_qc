import pandas as pd
import config
import numpy as np
from utils.logger import logging
from sklearn.preprocessing import LabelEncoder

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

check_missing_values()
    Identify and summarise missing values across QC features.

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

    gc_zero = data_frame[data_frame["picard_gc_dropout"] == 0]

    gc_zero_summary = (
        gc_zero
        .groupby(["sequencer", "cancer_type"])
        .size()
        .reset_index(name="zero_count")
    )

    logging.info(f"GC Dropout by sequencer and cancer type : {gc_zero_summary}")


if __name__ == "__main__":
    qc_summary_df = load_data(config.SUMMARY_QC_METRICS)
    explore_qc_data(qc_summary_df)

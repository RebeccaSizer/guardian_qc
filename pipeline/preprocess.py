import pandas as pd
import config
import numpy as np
from utils.logger import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import joblib
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

# Define the model features 
MODEL_FEATURES = [
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
    "picard_fold80",
    "picard_mean_target_coverage",
    "picard_median_target_coverage",
    "picard_target_bases_20x",
    "picard_target_bases_30x",
    "picard_target_bases_50x",
    "picard_target_bases_100x",
    "fastqc_duplication_rate",
    "fastp_duplication_rate",
    "fastqc_basic_status_pass|pass",
    "fastqc_basic_status_pass|pass|pass|pass"
]

CATERGORICAL_COLUMNS = ['fastqc_basic_status']
METADATA_COLUMNS = ['sample_name', 'cancer_type', 'sequencer']
STRATIFY_COLUMNS = ['cancer_type', 'sequencer']

# Load and audit the raw qc summary data
def load_data(file_path):

    summary_qc_df = pd.read_csv(file_path, header=0, sep="\t")
    logging.info(f"Loaded {summary_qc_df.shape[0]} rows x {summary_qc_df.shape[1]} columns from {file_path}")
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
    sep = "=" * 40
    sep_sub = '-' * 40
    logging.info(sep)
    logging.info("DATA SUMMARY")
    logging.info(sep)
    logging.info(f"Number of Rows: {data_frame.shape[0]}")
    logging.info(f"Number of Columns: {data_frame.shape[1]}")
    logging.info(f"Columns to list:\n {data_frame.columns.tolist()}")
    logging.info(f"\nSplit by sequencer and cancer type:\n {data_frame[STRATIFY_COLUMNS].value_counts()}")
    logging.info(sep)
    logging.info("DATA QUALITY ")
    logging.info(sep)
    logging.info(f"Duplicte rows: {data_frame.duplicated().sum()} / {data_frame.shape[0]}")
    logging.info(f"Missing values per column:\n{data_frame.isnull().sum()}")
    logging.info(f"Dtypes:\n{data_frame.dtypes}")
    logging.info(f"\nDescriptive stats:\n{data_frame.describe()}")

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

def fix_data_types(data_frame):
    """
    This fixes known issues with the data 
    before encoding or scaling """

    n_bad = (data_frame['picard_fold80'] == '?').sum()
    logging.info(f"picard_fold80: replacing {n_bad} '?' values with NaN")
    df = data_frame.copy()
    df['picard_fold80'] = pd.to_numeric(df["picard_fold80"].replace("?", np.nan))
    return df

def split_test_train(data_frame): # First iteration is using all the data without separating out the sequencer or cancertype 

    train_df, test_df = train_test_split(
        data_frame,
        test_size = 0.2,
        random_state = 42,
        stratify=data_frame[STRATIFY_COLUMNS] # Use cancer_type and sequencer to define training pop. These will NOT be given as features 
    )

    logging.info(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
    return train_df, test_df

# Encode catergorical data 
def fit_encoder(train_df: pd.DataFrame) -> OneHotEncoder:

    ohe = OneHotEncoder(categories='auto', sparse_output=False, handle_unknown='ignore')
    ohe.fit(train_df[CATERGORICAL_COLUMNS])
    logging.info(f"OHE fitted. Categories: {ohe.categories_}")
    return ohe

def apply_encoder(df: pd.DataFrame, ohe: OneHotEncoder) -> pd.DataFrame:

    encoded = ohe.transform(df[CATERGORICAL_COLUMNS])

    encoded_df = pd.DataFrame(
        encoded,
        columns=ohe.get_feature_names_out(CATERGORICAL_COLUMNS),
        index=df.index
    )

    data_frame = pd.concat(
        [df.drop(columns=CATERGORICAL_COLUMNS), encoded_df],
        axis=1
    )

    return data_frame

# Impute missing values from data 

def fit_imputer(train_df: pd.DataFrame) -> SimpleImputer:

    imputer = SimpleImputer(strategy = 'median')
    imputer.fit(train_df[MODEL_FEATURES])
    logging.info('Imputer fitted on training data')
    return imputer

def apply_imputer(df: pd.DataFrame, imputer: SimpleImputer) -> pd.DataFrame:

    imputed = imputer.transform(df[MODEL_FEATURES])
    imputed_df = pd.DataFrame(imputed, columns=MODEL_FEATURES, index=df.index)

    return imputed_df


# Scale the data
def fit_standard_scaler(train_df: pd.DataFrame):

    stdsc = StandardScaler()
    stdsc.fit(train_df)
    logging.info("Scaler fitted on training data.")
    return stdsc
    
def apply_standard_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:

    scaled = scaler.transform(df[MODEL_FEATURES])
    scaled_df = pd.DataFrame(scaled, columns=df.columns, index=df.index)
    return scaled_df

def save_transformers(ohe, imputer, scaler, out_dir: str) -> None:
    joblib.dump(ohe,     f"{out_dir}/ohe.pkl")
    joblib.dump(imputer, f"{out_dir}/imputer.pkl")
    joblib.dump(scaler,  f"{out_dir}/scaler.pkl")
    logging.info(f"Transformers saved to {out_dir}/")

def run_preprocessing(file_path: str, out_dir: str):

    # Load the input
    df = load_data(file_path)

    # Audit the data
    explore_qc_data(df)

    # Clean the data
    df = remove_duplicates(df)
    df = fix_data_types(df)

    # Split the data. This needs to be done before fitting anything 
    train_df, test_df = split_test_train(df)

    # Encode the catergorical values 
    # Fit on train only
    ohe = fit_encoder(train_df)
    train_df = apply_encoder(train_df, ohe)
    test_df = apply_encoder(test_df, ohe)

    # Drop the metadata for imputing and scaling 
    X_train = train_df[MODEL_FEATURES]
    X_test = test_df[MODEL_FEATURES]

    # Impute the data
    imputer = fit_imputer(X_train)
    X_train = apply_imputer(X_train, imputer)
    X_test = apply_imputer(X_test, imputer)

    # Scale the data
    scaler = fit_standard_scaler(X_train)
    X_train = apply_standard_scaler(X_train, scaler)
    X_test = apply_standard_scaler(X_test, scaler)

    # Save the transformers
    save_transformers(ohe, imputer, scaler, out_dir)

    logging.info(f"Preprocessing complete. X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, X_test


if __name__ == "__main__":
    X_train, X_test = run_preprocessing(
        file_path=config.SUMMARY_QC_METRICS,
        out_dir=config.PREPROCESSING_OUTDIR,
    )
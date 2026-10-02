import pandas as pd
import config
import numpy as np
import os 
from utils.logger import logging
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from outputs.graphs.preprocessing.preprocessing_graphs import plot_correlation_matrix,   plot_feature_distributions
import joblib
from pathlib import Path
import argparse
"""
guardian_qc preprocessing.py

Preprocessing utilities for the Guardian-QC project.

This module prepares QC summary data for downstream exploratory analysis,
dimensionality reduction and unsupervised machine-learning models such as
Isolation Forest.

The preprocessing workflow includes:

1. Data loading
    - Load QC summary data from a tab-separated file.
    - Return the data as a pandas DataFrame.

2. Data cleaning
    - Remove duplicate samples.
    - Convert QC metrics to appropriate numerical data types.
    - Replace invalid or missing values with NaN where appropriate.
    - Create an assay identifier from sequencer and cancer type.

3. Train/test splitting
    - Split the dataset into training and test sets.
    - Where appropriate, stratify the split using assay information to
      maintain representation of different sequencing/cancer-type groups.
    - The training set is used to fit preprocessing transformers to avoid
      data leakage into the test set.

4. Categorical encoding
    - Encode categorical QC variables using OneHotEncoder.
    - Handle previously unseen categories using ``handle_unknown="ignore"``.
    - Convert encoded categorical variables into numerical feature columns.

5. Missing-value imputation
    - Fit a median-based SimpleImputer using the training data only.
    - Apply the fitted imputer to both training and test data.
    - This ensures that information from the test set is not used when
      estimating missing-value replacement values.

6. Feature scaling
    - Fit a StandardScaler using the training data only.
    - Apply the fitted scaler to both training and test data.
    - Scaling is performed after categorical encoding and missing-value
      imputation so that the final feature matrix contains numerical values.

7. Feature selection
    - Remove features with zero or very low variance using VarianceThreshold.
    - Identify highly correlated features to support feature-selection
      decisions and reduce redundant information.

8. Transformer persistence
    - Save fitted preprocessing transformers using joblib.
    - Saved transformers include the one-hot encoder, imputer and scaler.
    - These can be reused to ensure that future data is processed using
      the same transformations as the training data.

The intended workflow is:

    Raw QC data
        |
        v
    Load data
        |
        v
    Remove duplicates
        |
        v
    Correct data types
        |
        v
    Create assay information
        |
        v
    Train/test split
        |
        v
    One-hot encoding
        |
        v
    Missing-value imputation
        |
        v
    Feature scaling
        |
        v
    Variance filtering
        |
        v
    Correlation assessment
        |
        v
    Model-ready feature matrix


Functions
---------

load_data()
    Load QC summary data from a tab-separated file.

explore_qc_data()
    Explores the raw qc metrics for missing, duplicate or incorrect values

remove_duplicates()
    Remove duplicate observations based on sample information.

fix_data_types()
    Convert QC metrics to appropriate numerical data types and create
    assay identifiers from sequencing and cancer-type information.
    
separate_data()
    Separates data based on Assay type and capture version

split_test_train()
    Split the input dataset into training and test sets, using stratification
    where appropriate.

fit_encoder()
    Fit a OneHotEncoder using the training data.

apply_encoder()
    Apply a fitted OneHotEncoder and return the encoded feature matrix.

fit_imputer()
    Fit a median-based SimpleImputer using training data.

apply_imputer()
    Apply a fitted imputer to a DataFrame.

fit_standard_scaler()
    Fit a StandardScaler using training data.

apply_standard_scaler()
    Apply a fitted StandardScaler to a DataFrame.

fit_robust_scaler()
    Fit a RobustScaler using training data.

apply_robust_scaler()
    Apply a fitted RobustScaler to a DataFrame.

save_transformers()
    Save fitted preprocessing transformers to disk using joblib.

fit_variance_threshold()
    Fit a VarianceThreshold feature selector.

apply_variance_threshold()
    Apply a fitted variance threshold selector.

find_correlated_features()
    Identify pairs of highly correlated features above a specified
    correlation threshold.

run_preprocessing()
    Run the complete preprocessing workflow and return the processed
    training and test feature matrices.


Data leakage prevention
-----------------------
Preprocessing transformers are fitted using the training data only.
The fitted transformers are then applied to the test data.

This prevents information from the test set influencing the preprocessing
parameters used during model development.


Date: 2026-08-18
Author: Rebecca Sizer
"""

# Load and audit the raw qc summary data
def load_data(file_path):
    """
    Loads the raw summary qc file containing all of the qc metrics.
    
    params:
        DataFrame containing qc metrics
        
    output:"""

    try:

        summary_qc_df = pd.read_csv(file_path, header=0, sep="\t")
        logging.info(f"Loaded {summary_qc_df.shape[0]} rows x {summary_qc_df.shape[1]} columns from {file_path}")
        return summary_qc_df

    except FileNotFoundError as e:
        logging.error(f"Sample QC file not found. Error: {e}")
        raise

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

    try:
        logging.info(sep)
        logging.info("DATA SUMMARY")
        logging.info(sep)
        logging.info(f"Number of Rows: {data_frame.shape[0]}")
        logging.info(f"Number of Columns: {data_frame.shape[1]}")
        logging.info(f"Columns to list:\n {data_frame.columns.tolist()}")
        logging.info(f"\nSplit by sequencer and cancer type:\n {data_frame[config.STRATIFY_COLUMNS].value_counts()}")
        logging.info(sep)
        logging.info("DATA QUALITY ")
        logging.info(sep)
        logging.info(f"Duplicte rows: {data_frame.duplicated().sum()} / {data_frame.shape[0]}")
        logging.info(f"Missing values per column:\n{data_frame.isnull().sum()}")
        logging.info(f"Dtypes:\n{data_frame.dtypes}")
        logging.info(f"\nDescriptive stats:\n{data_frame.describe()}")

    except Exception as e:
        logging.error(f"Error loading dataframe for preprocessing summary")
        raise


# Remove duplicate values from the Sample QC dataframe
def remove_duplicates(data_frame):
    """
    This function removes rows where the whole row is 
    duplicated elsewhere in the dataframe
    
    params:
        dataframe: containing qc summary metrics
        
    output
        dataframe: with duplicate rows removed 
    """
    try:
        logging.info(f"Removing any duplicate rows...") # Should I remove duplicate sample names - will this bias the model
        original_len = len(data_frame)
        deduplicated_df = data_frame.drop_duplicates()
        final_len = len(deduplicated_df)
        logging.info(f"Total rows removed due to duplication: {original_len - final_len}")

        return deduplicated_df

    except Exception as e:
        logging.error(f"Error when removing duplicates: {e}")
        raise


def fix_data_types(data_frame):
    """
    This fixes known issues with the data 
    before encoding or scaling.
    
        - Remove '?' values from the Fold80 column
        - Replace with NaN
    
    params:
        dataframe containing QC metrics including picard_fold80 columns
        
    output:
        dataframe with cleaned picard_fold80 column"""

    try:
        logging.info('Removing rows with missing values')
        data_frame = data_frame[data_frame['picard_fold80'] != '?']
        return data_frame

    except KeyError as e:
        logging.error(f"Missing column (picard_fold80): {e}")
        raise


def separate_data(df: pd.DataFrame, assay_type: str, version: str) -> pd.DataFrame:
    """
    Filter the QC dataset to the current production assay. Only data from the NovaSeqX will be used
    from here on.

    ST:
        NovaSeq X + solid_tumour_v3

    Haem:
        NovaSeq X + haem_v3

    params
        df : pd.DataFrame
            Cleaned QC dataframe.

        assay_type : str
            Either "ST" or "haem".

        version : str
            Either v2, v3 or v2_v3 for both 

    output:
        pd.DataFrame containing only the requested assay.
    """

    assay_type = assay_type.lower()

    try:
        if assay_type == "st" and version == "v3":
            filtered_df = df[
                (df["sequencer"].str.lower() == "novaseqx") &
                (df["cancer_type"].str.lower() == "solid_tumour_v3")
            ].copy()

        elif assay_type == "haem" and version == "v3":
            filtered_df = df[
                (df["sequencer"].str.lower() == "novaseqx") &
                (df["cancer_type"].str.lower() == "haem_v3")
            ].copy()

        elif assay_type == "st" and version == "v2":
            filtered_df = df[
                        (df["sequencer"].str.lower() == "novaseqx") &
                        (df["cancer_type"].str.lower() == "solid_tumour")
                    ].copy()

        elif assay_type == "haem" and version == "v2":
            filtered_df = df[
                        (df["sequencer"].str.lower() == "novaseqx") &
                        (df["cancer_type"].str.lower() == "haem_v2")
                    ].copy()

        elif assay_type == "st" and version == "v2_v3":
            filtered_df = df[
                        (df["sequencer"].str.lower() == "novaseqx") &
                        (df["cancer_type"].str.lower().isin(["solid_tumour", "solid_tumour_v3"]))
                    ].copy()

        elif assay_type == "haem" and version == "v2_v3":
            filtered_df = df[
                        (df["sequencer"].str.lower() == "novaseqx") &
                        (df["cancer_type"].str.lower().isin(["haem_v2", "haem_v3"]))
                    ].copy()

        else:
            raise ValueError(
                f"Unknown assay_type '{assay_type}'. "
                "Expected 'ST' or 'haem'."
            )

        logging.info(
            f"Filtering for {assay_type.upper()} assay: "
            f"{len(filtered_df)} samples retained"
        )

        logging.info(
            f"Sequencers:\n{filtered_df['sequencer'].value_counts().to_string()}"
        )

        logging.info(
            f"Cancer/pipeline versions:\n"
            f"{filtered_df['cancer_type'].value_counts().to_string()}"
        )
        filtered_df.to_csv(f"data/processed/split_data_only_{assay_type}_{version}.csv")
        return filtered_df


    except Exception as e:
        logging.error(f"There was an error when splitting the data: {e}")
        raise


def split_test_train(data_frame):
    """ 
    This function splits the data into the test and training set.
    params:
        dataframe containing qc metrics

    output:
        Training dataframe
        Testing dataframe
    """
    try:
        train_df, test_df = train_test_split(
            data_frame,
            test_size = 0.2,
            random_state = 42,
            stratify=data_frame[config.STRATIFY_COLUMNS] # Use cancer_type and sequencer to define training pop. These will NOT be given as features 
        )

        logging.info(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")
        return train_df, test_df

    except Exception as e:
        logging.error(f"Error when splitting the data into train and test datasets: {e}")
        raise


# Encode catergorical data 
def fit_encoder(train_df: pd.DataFrame) -> OneHotEncoder:
    """
    This dunction fits the OneHotEncoder to the catergorical values
    in the training dataframe
    
    params:
        dataframe of qc values
        
    output:
        OneHotEncoder fit to the training data
    """
    try:
        ohe = OneHotEncoder(categories='auto', sparse_output=False, handle_unknown='ignore')
        ohe.fit(train_df[config.CATERGORICAL_COLUMNS])
        logging.info(f"OHE fitted. Categories: {ohe.categories_}")
        return ohe

    except Exception as e:
        logging.error(f"Error fitting the One Hot Encoder: {e}")
        raise 


def apply_encoder(df: pd.DataFrame, ohe: OneHotEncoder) -> pd.DataFrame:
    """
    This function fits the OneHotEncoder to the catergorical values
    in the training dataframe
    
    params:
        dataframe of qc values
        OneHotEncoder trained on the training data
        
    output:
        dataframe with catergorical values encoded
    """
    try:
        encoded = ohe.transform(df[config.CATERGORICAL_COLUMNS])

        encoded_df = pd.DataFrame(
            encoded,
            columns=ohe.get_feature_names_out(config.CATERGORICAL_COLUMNS),
            index=df.index
        )

        data_frame = pd.concat(
            [df.drop(columns=config.CATERGORICAL_COLUMNS), encoded_df],
            axis=1
        )

        return data_frame

    except Exception as e:
        logging.error(f"Error applying One Hot Encoder to data: {e}")
        raise


# Impute missing values from data 
def fit_imputer(train_df: pd.DataFrame) -> SimpleImputer:
    """
    This function fits the SimpleImputer to the numerical columns
    in the training dataframe
    
    params:
        dataframe of qc values
        
    output:
        SimpleImputer trained on the training data
    """
    try:
        imputer = SimpleImputer(strategy = 'mean')
        imputer.fit(train_df)
        logging.info('Imputer fitted on training data')
        return imputer

    except Exception as e:
        logging.error(f"Error when fitting the imputer: {e}")
        raise


def apply_imputer(df: pd.DataFrame, imputer: SimpleImputer) -> pd.DataFrame:
    """
    This function fits the SimpleImputer to the numerical values
    in the training dataframe
    
    params:
        dataframe of qc values
        SimpleImputer trained on the training data
        
    output:
        dataframe with all missing values replaced by the median
    """
    try:
        imputed = imputer.transform(df)
        imputed_df = pd.DataFrame(imputed, columns=df.columns, index=df.index)

        return imputed_df

    except Exception as e:
        logging.error(f"Error applying imputer: {e}")
    raise


# Scale the data: Use standard scaler first, 
# But it may be worth testing the robust scaler at some point.
def fit_standard_scaler(train_df: pd.DataFrame):
    """
    This function fits the StandardScaler to the numerical columns
    in the training dataframe
    
    params:
        dataframe of qc values
        
    output:
        StandardScaler trained on the training data
    """
    try:
        stdsc = StandardScaler()
        stdsc.fit(train_df)
        logging.info("Scaler fitted on training data.")
        return stdsc

    except Exception as e:
        logging.error(f"Error what fitting the standard scaler: {e}")
    
def apply_standard_scaler(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:

    """
    This function fits the StandardScaler to the numerical columns
    in the training dataframe
    
    params:
        dataframe of qc values
        
    output:
        dataframe with scaled numerical values 
    """
    try:
        scaled = scaler.transform(df)
        scaled_df = pd.DataFrame(scaled, columns=df.columns, index=df.index)
        return scaled_df
    except Exception as e:
        logging.error(f"Error when applying the standard scaler: {e}")


# Robust scaler for future use 
def fit_robust_scaler(train_df: pd.DataFrame):

    """
    This function fits the RobustScaler to the numerical columns
    in the training dataframe
    
    params:
        dataframe of qc values
        
    output:
        StandardScaler trained on the training data
    """
    rbssc = RobustScaler()
    rbssc.fit(train_df)
    logging.info("Scaler fitted on training data.")
    return rbssc
    
def apply_robust_scaler(df: pd.DataFrame, scaler: RobustScaler) -> pd.DataFrame:
    """
    This function fits the RobustScaler to the numerical columns
    in the training dataframe
    
    params:
        dataframe of qc values
        
    output:
        dataframe with scaled numerical values 
    """
    scaled = scaler.transform(df)
    scaled_df = pd.DataFrame(scaled, columns=df.columns, index=df.index)
    return scaled_df


# Save the transformers
def save_transformers(ohe, imputer, scaler, vt, to_drop_columns, out_dir: str) -> None:
    """
    This function saves the .pkl files so that this preprocessing can
    be applied to future data.
    
    params:
        ohe: Fitted OneHotEncoder
        imputer: Fitted SimpleImputer
        scaler: Fitted StandardScaler
        out_dit: Output directory
        
    output:
        ohe, imputer, and scaler .pkl files saved to the specified directory
    """
    try:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(ohe,     f"{out_dir}/ohe.pkl")
        joblib.dump(imputer, f"{out_dir}/imputer.pkl")
        joblib.dump(scaler,  f"{out_dir}/scaler.pkl")
        joblib.dump(vt,  f"{out_dir}/vt.pkl")
        joblib.dump(to_drop_columns,  f"{out_dir}/dropped_columns.pkl")
        logging.info(f"Transformers saved to {out_dir}/")

    except Exception as e:
        logging.error(f"Error saving transformers: {e}")


# Feature selection
# Identify columns that have very little variance
def fit_variance_threshold(X_train: pd.DataFrame, threshold:float = 0.01) -> VarianceThreshold:
    """
    This function sets the VarianceThreshold
    
    params:
        dataframe of qc values
        threshold: set to 0.01 unless specified otherwise 
        
    output:
        VarianceThreshold fitted on the training data
    """
    try:
        vt = VarianceThreshold(threshold=threshold)
        vt.fit(X_train)
        dropped = X_train.columns[~vt.get_support()].tolist()
        logging.info(f"VarianceThreshold dropping {len(dropped)} features: {dropped}")
        return vt

    except Exception as e:
        logging.error(f"Error fitting variance threshold: {e}")

def apply_variance_threshold(df: pd.DataFrame, vt: VarianceThreshold) -> pd.DataFrame:

    """
    This function fits the VarianceThreshold to the numerical columns
    in the training dataframe
    
    params:
        dataframe of qc values
        vt: the VarianceThreshold trained on the training data
        
    output:
        dataframe with variance failed returned 
    """
    try:
        df = pd.DataFrame(
            vt.transform(df),
            columns=df.columns[vt.get_support()],
            index=df.index
        )
        return df

    except Exception as e:
        logging.error(f"Error applying variance threshold: {e}")


# Find and Remove highly correlated pairs of data to prevent overweighting 
def find_correlated_features(X_train, threshold=0.95):
    """
    This function finds qc metrics that are highly correlated.
    Removing highly correlated features prevents over weighting those features.
    
    params:
        X_train: dataframe of the training data
        threshold: threshold for correlation
    
    output: 
        dataframe of correlated pairs
    """
    
    corr_matrix = X_train.corr().abs() #calculates pearson correlation coefficient between each pair 

    upper = corr_matrix.where(
        np.triu(        # only the upper triangle
            np.ones(corr_matrix.shape), #create a matrix of 1's with the same dimentions as corr_matrix
            k=1
        ).astype(bool) #this becomes a boolean mask 
    )

    correlated_pairs = []

    for column in upper.columns:
        for row in upper.index:
            correlation = upper.loc[row, column]

            if pd.notna(correlation) and correlation > threshold:
                correlated_pairs.append({
                    "feature_1": row,
                    "feature_2": column,
                    "correlation": correlation
                })

    return pd.DataFrame(correlated_pairs)



# Run preprocessing 
def run_preprocessing(file_path: str, assay: str, version: str, out_dir: None):
    """
    This function strings together all of the preprocessing steps
    to prepare the data for training the machine learning model.
    Preprocessing steps include removing duplicates,
    fixing data types, separate data by assay and version, split into 
    test and train, fit and apply a one-hot encoder, fit and apply as
    imputer, fit and apply a scaler, apply variance threshold to rmove metrics 
    with no variation, find and drop correlated features 
    
    params:
        file_path: path to the qc_summary metrics file 
        assay_type: haem or st
        out_dir: where to save the transformers if not None
        version: capture version
    
    output:
        X_train: The train split that has been preprocessed 
        X_test: The test split that has been preprocessed
        train_meta: Meta data associated with the training dataset
        test_meta: Meta data associated with the test dataset 
    """
    
    # Load the input
    try:
        df = load_data(file_path)
    except FileNotFoundError as e:
        logging.error(f"File not found for preprocessing: {e}")
        raise

    # Audit the data
    explore_qc_data(df)

    # Clean the data
    df = remove_duplicates(df)
    df = fix_data_types(df)

    # Filter to the requested production assay
    df = separate_data(
        df,
        assay_type=assay,
        version=version
    )

    if len(df) < 10:
        raise ValueError(
            f"Only {len(df)} samples available for {assay}. "
            "Not enough data for preprocessing."
        )

    # Split the data. This needs to be done before fitting anything 
    train_df, test_df = split_test_train(df)

    # Isolate the metadata to save 
    try:
        train_meta = train_df[["sample_name", "cancer_type", "sequencer", "assay_type", "worklist"]].copy()  # keep assay label for per-assay plots later
        test_meta  = test_df[["sample_name", "cancer_type", "sequencer", "assay_type", "worklist"]].copy()

    except KeyError as e:
        logging.info(f"Missing column from data for preprocessing: {e}")

    # Encode the catergorical values 
    # Fit on train only
    ohe = fit_encoder(train_df)
    train_df = apply_encoder(train_df, ohe)
    test_df = apply_encoder(test_df, ohe)

    # Build updated feature list post-OHE
    ohe_cols = list(ohe.get_feature_names_out(config.CATERGORICAL_COLUMNS))
    base_features = [f for f in config.MODEL_FEATURES
                     if not f.startswith("fastqc_basic_status_")]
    all_features = base_features + ohe_cols

    # Drop the metadata
    # Extract feature matrices — drop metadata
    try:
        X_train = train_df[all_features]
        X_test  = test_df[all_features]

    except KeyError as e:
        logging.error(f"Missing column from preprocessing data: {e}")

    # Impute the data
    logging.info('Fitting Imputer ...')
    imputer = fit_imputer(X_train)

    logging.info('Applying imputer ... ')
    X_train = apply_imputer(X_train, imputer)
    X_test = apply_imputer(X_test, imputer)

    # Scale the data
    scaler = fit_robust_scaler(X_train)

    logging.info('Scaling data using the Robust scaler') # Chose to use robust scaler as less affected by outliers. 
    X_train = apply_robust_scaler(X_train, scaler)
    X_test = apply_robust_scaler(X_test, scaler)

    # Variance threshold - Decided against dropping based on variance
    vt = fit_variance_threshold(X_train)
    X_train = apply_variance_threshold(X_train, vt)
    X_test = apply_variance_threshold(X_test, vt)

    # Remove correlated features to prevent too much weight on certian features 
    correlated = find_correlated_features(X_train)
    logging.info(f'Analysing correlation between features: {print(correlated)}')

    plot_correlation_matrix(X_train, assay, os.path.join('outputs/graphs/preprocessing/metric_correlation_matrix_all_columns/'))

    if assay.lower() == 'haem':
        columns_to_drop = [
            'bcftools_ts',
            'picard_pf_reads',
            'picard_mean_target_coverage',
            'picard_mean_insert',
            'fastqc_basic_status_pass|pass',
            'fastqc_basic_status_pass|pass|pass|pass',
            'bcftools_variants',
            'picard_target_bases_20x',
            'picard_target_bases_30x',
            'picard_target_bases_50x'
        ]
    elif assay.lower() == 'st':
        columns_to_drop = [
            'picard_total_reads',
            'picard_pf_reads',
            'picard_mean_target_coverage',
            'picard_mean_insert',
            'picard_median_target_coverage',
            'fastqc_basic_status_pass|pass',
            'fastqc_basic_status_pass|pass|pass|pass',
            'bcftools_variants',
            'picard_target_bases_20x',
            'picard_target_bases_30x',
            'picard_target_bases_50x'

        ]

    # Only drop columns that are actually present
    to_drop = [
        column for column in columns_to_drop
        if column in X_train.columns
    ]

    logging.info(
        f"Dropping {len(to_drop)} columns due to high correlation: {to_drop}"
    )

    logging.info(f'Dropping {len(to_drop)} columns due to high correlation: {to_drop}')

    X_train = X_train.drop(columns=to_drop)
    X_test = X_test.drop(columns=to_drop)

    plot_correlation_matrix(X_train, assay, os.path.join('outputs/graphs/preprocessing/metric_correlation_matrix_filtered/'))

    if out_dir is not None:
        save_transformers(ohe, imputer, scaler, vt, to_drop, out_dir)

    logging.info(f"Preprocessing complete. X_train: {X_train.shape} | X_test: {X_test.shape}")
    return X_train, X_test, train_meta, test_meta


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Preprocess Guardian-QC data for a specific assay."
    )

    parser.add_argument(
        "--assay",
        choices=["st", "haem"],
        required=True,
        help="Assay to preprocess: ST or haem"
    )

    parser.add_argument(
        "--version",
        choices=["v2", "v3", "v2_v3"],
        required=True,
        help="Capture version for preprocess: v2, v3 or v2_v3"
    )

    args = parser.parse_args()

    output_dir = os.path.join(
        config.PREPROCESSING_OUTDIR,
        args.assay
    )

    X_train, X_test, train_meta, test_meta = run_preprocessing(
        file_path=config.SUMMARY_QC_METRICS,
        assay=args.assay,
        version=args.version,
        out_dir=output_dir,
    )

    logging.info(
        f"{args.assay} preprocessing complete."
    )

    print(
        f"Training samples: {len(X_train)}, meta data: {len(train_meta)}"
    )

    print(
        f"Testing samples: {len(X_test)}, meta data: {len(test_meta)}"
    )
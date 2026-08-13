import pandas as pd
import numpy as np
from utils.logger import logging
from sklearn.preprocessing import LabelEncoder

"""
Preprocessing the data includes:
    Handling missing values
    Removing duplicates
    Correcting data types
    Scaling/normalising numerical features
    Encoding categorical variables
    Removing irrelevant columns

Also complete feature engineering 
    Creating error_rate_per_1000_reads
    Combining SNVs + indels into a total variant count
    Creating ratios such as Ti/Tv
    Extracting sequencer from a run identifier
    Creating a feature such as coverage_pass_rate
    Log-transforming a highly skewed QC metric
"""

def preprocess_qc_metrics(df):

    logging.info(f"Loaded: {df.shape[0]} patients, {df.shape[1]} qc metrics")
    logging.info(f"sequencer distribution:\n{df['sequencer'].value_counts()}")
    logging.info(f"Missing values: {df.isnull().sum().sum()}")

    le = LabelEncoder()
    df["fastqc_basic_status_encoded"] = le.fit_transform(df['fastqc_basic_status'])
    fastqc_basic_status_values = le.classes_
    logging.info(f"Classes: {fastqc_basic_status_values}")
    logging.info(f"Column headers: {print(df)}")

    return df

import pandas as pd
import numpy as np
from utils.logger import logging
from sklearn.preprocessing import LabelEncoder

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

import os
import joblib
import numpy as np
import pandas as pd
import pytest

from pathlib import Path
from sklearn.ensemble import IsolationForest

import pipeline.train_composite as model

@pytest.fixture # make a fixture to generate synthetic data for testing
def model_data():

    np.random.seed(42) # Set a random seed for reproducibility

    X_train = pd.DataFrame({
        "metric_a": np.random.normal(100, 5, 100), # Generate 100 samples from a normal distribution with mean=100 and std=5
        "metric_b": np.random.normal(50, 3, 100), # Generate 100 samples from a normal distribution with mean=50 and std=3
        "metric_c": np.random.normal(200, 10, 100) # Generate 100 samples from a normal distribution with mean=200 and std=10
    })

    X_test = pd.DataFrame({
        "metric_a": [100, 102, 98, 200, 95], # Generate 5 test samples, including an outlier (200)
        "metric_b": [50, 51, 49, 150, 48], # Generate 5 test samples, including an outlier (150)
        "metric_c": [200, 205, 195, 500, 190] # Generate 5 test samples, including an outlier (500)
    })

    return X_train, X_test


def test_fit_isolation_forest(model_data):

    X_train, _ = model_data

    fitted_model = model.fit_isolation_forest(
        X_train=X_train,
        assay="haem",
        version="v2",
        contamination=0.08,
        n_estimators=100,
        max_samples=50,
        max_features=1.0
    )

    assert isinstance(fitted_model, IsolationForest) # Check that the returned model is an instance of IsolationForest

    assert fitted_model.n_estimators == 100 # Check that the number of estimators is set correctly
    assert fitted_model.contamination == 0.08
    assert fitted_model.max_samples == 50
    assert fitted_model.max_features == 1.0

    # Check that the model has been fitted
    assert hasattr(fitted_model, "estimators_") # Check that the model has been fitted and has estimators_ attribute


def test_save_model(model_data, tmp_path):

    X_train, _ = model_data

    fitted_model = model.fit_isolation_forest(
        X_train, "haem", "v2",
        0.08, 100, 50, 1.0
    )

    output_dir = tmp_path / "models"

    model.save_model(
        fitted_model,
        "haem",
        "v2",
        0.08,
        100,
        50,
        1.0,
        str(output_dir)
    )

    expected_file = (
        output_dir /
        "haem_v2_c0.08_n100_ms50_mf1.0_isolation_forest.pkl"
    )

    assert expected_file.exists() # check that the model file was created
    assert expected_file.stat().st_size > 0 # check that the model file is not empty


def test_load_model(model_data, tmp_path):

    X_train, _ = model_data

    fitted_model = model.fit_isolation_forest(
        X_train, "haem", "v2",
        0.08, 100, 50, 1.0
    )

    output_dir = tmp_path / "models"

    model.save_model(
        fitted_model,
        "haem",
        "v2",
        0.08,
        100,
        50,
        1.0,
        str(output_dir)
    )

    loaded_model = model.load_model(
        "haem",
        "v2",
        0.08,
        100,
        50,
        1.0,
        str(output_dir)
    )

    assert isinstance(loaded_model, IsolationForest) # Check that the loaded model is an instance of IsolationForest

    # Check that the loaded model produces the same predictions
    np.testing.assert_array_equal(
        fitted_model.predict(X_train),
        loaded_model.predict(X_train)
    )


def test_load_model_file_not_found(tmp_path):

    with pytest.raises(FileNotFoundError): # Check that loading a non-existent model raises a FileNotFoundError

        model.load_model(
            "haem",
            "v2",
            0.08,
            100,
            50,
            1.0,
            str(tmp_path)
        )


def test_score_samples(model_data):

    X_train, X_test = model_data

    original_test = X_test.copy()

    fitted_model = model.fit_isolation_forest(
        X_train, "haem", "v2",
        0.08, 100, 50, 1.0
    )

    result = model.score_samples(
        fitted_model,
        X_test,
        "haem",
        "v2",
        "test"
    )

    assert isinstance(result, pd.DataFrame) # Check that the result is a DataFrame

    assert "anomaly_score" in result.columns # Check that the result DataFrame contains the "anomaly_score" column
    assert "anomaly_label" in result.columns # Check that the result DataFrame contains the "anomaly_label" column

    assert len(result) == len(X_test) # Check that the number of rows in the result DataFrame matches the number of rows in the test DataFrame

    assert set(result["anomaly_label"].unique()).issubset({-1, 1}) # Check that the anomaly labels are either -1 (outlier) or 1 (normal)

    assert np.isfinite(result["anomaly_score"]).all() # Check that all anomaly scores are finite numbers

    # Original DataFrame should remain unchanged
    pd.testing.assert_frame_equal(X_test, original_test) # Check that the original test DataFrame remains unchanged after scoring

    # Original feature columns should be retained
    assert list(result.columns[:3]) == list(X_test.columns) # Check that the first three columns of the result DataFrame match the original feature columns in the test DataFrame


def test_explain_outlier():

    X_train = pd.DataFrame({
        "metric_a": [10, 11, 10, 12, 11],
        "metric_b": [20, 21, 20, 22, 21],
        "metric_c": [30, 31, 30, 32, 31]
    })

    sample = pd.Series({
        "metric_a": 100,
        "metric_b": 21,
        "metric_c": 31,
        "anomaly_score": -0.2,
        "anomaly_label": -1
    })

    result = model.explain_outlier(
        sample,
        X_train,
        top_n=2
    )

    assert isinstance(result, pd.DataFrame) # Check that the result is a DataFrame

    assert len(result) == 2 # Check that the number of rows in the result DataFrame matches the top_n parameter

    expected_columns = [
        "feature",
        "sample_value",
        "train_median",
        "mad",
        "robust_z_score"
    ]

    assert list(result.columns) == expected_columns # Check that the columns in the result DataFrame match the expected columns

    # The deliberately extreme metric should rank first
    assert result.iloc[0]["feature"] == "metric_a" # Check that the feature with the highest deviation is ranked first

    # Explanations should be ranked by deviation
    assert result["robust_z_score"].is_monotonic_decreasing # Check that the robust z-scores are ranked in descending order


def test_explain_all_outliers(tmp_path):

    X_train = pd.DataFrame({
        "metric_a": [10, 11, 10, 12, 11],
        "metric_b": [20, 21, 20, 22, 21]
    })

    X_scored = pd.DataFrame({
        "sample_name": ["sample1", "sample2", "sample3"],
        "metric_a": [100, 90, 11],
        "metric_b": [20, 100, 21],
        "anomaly_score": [-0.2, -0.3, 0.1],
        "anomaly_label": [-1, -1, 1]
    })

    result = model.explain_all_outliers(
        X_scored,
        X_train,
        "haem",
        "v2",
        str(tmp_path),
        top_n=2
    )

    assert isinstance(result, pd.DataFrame) # Check that the result is a DataFrame

    assert len(result) == 2 # Check that the number of rows in the result DataFrame matches the top_n parameter

    assert "sample_name" in result.columns # Check that the result DataFrame contains the "sample_name" column
    assert "top_1_feature" in result.columns   # Check that the result DataFrame contains the "top_1_feature" column
    assert "top_1_robust_z_score" in result.columns # Check that the result DataFrame contains the "top_1_robust_z_score" column
    assert set(result["sample_name"]) == {"sample1", "sample2"} # Check that the result DataFrame contains only the outlier samples


def test_explain_all_outliers_no_anomalies(tmp_path):

    X_train = pd.DataFrame({
        "metric_a": [10, 11, 10],
        "metric_b": [20, 21, 20]
    })

    X_scored = pd.DataFrame({
        "sample_name": ["sample1", "sample2"],
        "metric_a": [10, 11],
        "metric_b": [20, 21],
        "anomaly_score": [0.1, 0.2],
        "anomaly_label": [1, 1]
    })

    result = model.explain_all_outliers(
        X_scored,
        X_train,
        "haem",
        "v2",
        str(tmp_path)
    )

    assert isinstance(result, pd.DataFrame) # Check that the result is a DataFrame
    assert result.empty # Check that the result DataFrame is empty since there are no anomalies in the input data

def test_attach_metadata():

    X_scored = pd.DataFrame({
        "metric_a": [10, 20],
        "anomaly_score": [-0.2, 0.1],
        "anomaly_label": [-1, 1]
    }, index=[0, 1])

    meta = pd.DataFrame({
        "sample_name": ["sample1", "sample2"],
        "assay_type": ["haem", "haem"]
    }, index=[0, 1])

    result = model.attach_metadata(X_scored, meta)

    assert isinstance(result, pd.DataFrame)

    # Check that the metadata columns are correctly attached to the scored DataFrame
    assert "sample_name" in result.columns
    assert "assay_type" in result.columns
    assert list(result.columns[:2]) == [
        "sample_name", "assay_type"
    ]

    # Check that the original scored DataFrame columns are retained
    assert "anomaly_score" in result.columns
    assert "anomaly_label" in result.columns

    # Check that the number of rows in the result DataFrame matches the number of rows in the scored DataFrame
    assert len(result) == 2


def test_log_model_summary(caplog): # Test that the log_model_summary function logs the expected information

    import logging

    with caplog.at_level(logging.INFO):

        model.log_model_summary(
            assay="haem",
            version="v2",
            n_train=100,
            n_test=25,
            n_flagged_train=8,
            n_flagged_test=3,
            contamination=0.08,
            auprc=0.75
        )

    assert "MODEL SUMMARY" in caplog.text
    assert "haem_v2" in caplog.text
    assert "Training samples:" in caplog.text
    assert "Test samples:" in caplog.text
    assert "AUPRC (labelled eval):" in caplog.text


def test_explain_outliers_shap(tmp_path, monkeypatch):

    X_scored = pd.DataFrame({
        "metric_a": [10, 100, 12],
        "metric_b": [20, 200, 22],
        "anomaly_score": [0.1, -0.2, 0.2],
        "anomaly_label": [1, -1, 1]
    })

    features = ["metric_a", "metric_b"]

    # Fake SHAP explanation
    class FakeExplanation:

        def __init__(self, values):
            self.values = values

    class FakeExplainer:

        def __init__(self, fitted_model):
            pass

        def __call__(self, X):
            return FakeExplanation(
                np.array([[0.8, 0.2]])
            )

    monkeypatch.setattr(
        model.shap,
        "TreeExplainer",
        FakeExplainer
    )

    result = model.explain_outliers_shap(
        X_scored,
        model=None,
        features=features,
        out_dir=str(tmp_path),
        top_n=2
    )

    assert isinstance(result, pd.DataFrame)

    # Check SHAP explanation columns
    assert "top_1_metric" in result.columns
    assert "top_1_shap" in result.columns
    assert "top_1_abs_shap" in result.columns

    # Only the flagged sample should receive SHAP values
    assert pd.notna(result.loc[1, "top_1_metric"])

    assert pd.isna(result.loc[0, "top_1_metric"])
    assert pd.isna(result.loc[2, "top_1_metric"])

    # Largest absolute SHAP value should rank first
    assert result.loc[1, "top_1_metric"] == "metric_a"

# test the run_train_model function 
def test_run_model_train(
    model_data, tmp_path, monkeypatch
):

    X_train, X_test = model_data

    train_meta = pd.DataFrame({
        "sample_name": [f"train_{i}" for i in range(len(X_train))]
    })

    test_meta = pd.DataFrame({
        "sample_name": [f"test_{i}" for i in range(len(X_test))]
    })

    # Redirect model saving to temporary directory
    monkeypatch.setattr(
        model.config,
        "TRAINED_COMPOSITE_MODEL_OUTDIR",
        str(tmp_path / "models")
    )

    # Avoid running SHAP during this orchestration test
    monkeypatch.setattr(
        model,
        "explain_outliers_shap",
        lambda X_scored, fitted_model, features, out_dir, top_n: X_scored
    )

    # Ensure all features are available for this synthetic test
    monkeypatch.setattr(
        model.config,
        "COL_TO_DROP_HAEM",
        []
    )

    output_dir = tmp_path / "results"

    train_results, test_results = model.run_model_train(
        X_train,
        X_test,
        train_meta,
        test_meta,
        "haem",
        "v2",
        0.08,
        100,
        50,
        1.0,
        str(output_dir)
    )

    assert isinstance(train_results, pd.DataFrame)
    assert isinstance(test_results, pd.DataFrame)

    assert len(train_results) == len(X_train)
    assert len(test_results) == len(X_test)

    assert "sample_name" in train_results.columns
    assert "sample_name" in test_results.columns

    assert "anomaly_score" in train_results.columns
    assert "anomaly_label" in test_results.columns

    # Check output files
    assert len(list(output_dir.glob("*.csv"))) == 2

    # Check model saved in temporary directory
    assert len(list((tmp_path / "models").rglob("*.pkl"))) == 1
    

# Test the graph functions in train_composite.py

def test_plot_score_distribution(tmp_path):

    X_train_scored = pd.DataFrame({
        "anomaly_score": np.random.normal(0.1, 0.05, 30),
        "anomaly_label": [1] * 25 + [-1] * 5
    })

    X_test_scored = pd.DataFrame({
        "anomaly_score": np.random.normal(0.1, 0.05, 15),
        "anomaly_label": [1] * 12 + [-1] * 3
    })

    model.plot_score_distribution(
        X_train_scored,
        X_test_scored,
        "haem",
        str(tmp_path)
    )

    output_file = tmp_path / "haem_score_distribution.png"

    assert output_file.exists()
    assert output_file.stat().st_size > 0


def test_plot_top_deviant_features(tmp_path):

    explanations = pd.DataFrame({
        "top_1_feature": [
            "metric_a",
            "metric_a",
            "metric_b",
            "metric_c",
            "metric_a"
        ]
    })

    model.plot_top_deviant_features(
        explanations,
        "haem",
        str(tmp_path),
        top_n=3
    )

    output_file = tmp_path / "haem_top_deviant_features.png"

    assert output_file.exists() # Check that the output file was created
    assert output_file.stat().st_size > 0 # Check that the output file is not empty


def test_plot_top_deviant_features_empty(tmp_path):

    explanations = pd.DataFrame()

    model.plot_top_deviant_features(
        explanations,
        "haem",
        str(tmp_path)
    )

    output_file = tmp_path / "haem_top_deviant_features.png"

    assert not output_file.exists() # Check that the output file was not created since the explanations DataFrame is empty


def test_plot_anomaly_score(tmp_path, monkeypatch):

    monkeypatch.setattr(
        model.config,
        "TRAINING_PLOT_DIR",
        str(tmp_path)
    )

    X_scored = pd.DataFrame({
        "anomaly_score": [0.1, 0.2, -0.1, -0.3],
        "anomaly_label": [1, 1, -1, -1]
    })

    model.plot_anomaly_score(
        X_scored,
        "haem",
        "v2"
    )

    output_file = (
        tmp_path /
        "haem" /
        "v2" /
        "anomaly_score_scatter_haem_v2.png"
    )

    assert output_file.exists()
    assert output_file.stat().st_size > 0


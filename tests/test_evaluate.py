import pipeline.evaluate as evaluation
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

# set values for config.SHAP_COLUMNS and config.ROBUST_Z_COLUMNS to small, predictable lists
@pytest.fixture
def configured_columns(monkeypatch):
    """Use small, predictable explanation-column lists in the tests."""

    monkeypatch.setattr(
        evaluation.config, # change the config module in the evaluation module
        "SHAP_COLUMNS",
        ["shap_1", "shap_2"]
    )

    monkeypatch.setattr(
        evaluation.config,
        "ROBUST_Z_COLUMNS",
        ["robust_z_1", "robust_z_2"]
    )

# Create a dataframe with model and truth data for testing the evaluation functions. 
@pytest.fixture
def evaluation_files(tmp_path, configured_columns):
    """Create isolated model/truth CSV files for testing."""

    model = pd.DataFrame(
        {
            "sample_name": ["S1", "S2", "S3", "S4", "MODEL_ONLY"],

            # Model predictions:
            # S1 = TN, S2 = TP, S3 = FN, S4 = FP
            "anomaly_label": [1, -1, 1, -1, 1],

            # SHAP explanations only for predicted outliers
            "shap_1": [None, "fold80", None, "coverage", None],
            "shap_2": [None, "insert_size", None, None, None],

            # Robust Z-score explanations only for predicted outliers
            "robust_z_1": [None, "insert_size", None, "coverage", None],
            "robust_z_2": [None, None, None, None, None],

            "anomaly_score": [0.1, -0.3, 0.1, -0.5, 0.2],
        }
    )

    truth = pd.DataFrame(
        {
            "sample_name": [
                " S1 ", "S2", "S3", "S4", "TRUTH_ONLY"
            ],

            "anomaly_label": [1, -1, -1, 1, -1],

            "truth_feature_1": [
                None, "fold80", "coverage", None, "coverage"
            ],

            "truth_feature_2": [
                None, None, "insert_size", None, None
            ],
        }
    )

    model_path = tmp_path / "model.csv"
    truth_path = tmp_path / "truth.csv"

    model.to_csv(model_path, index=False)
    truth.to_csv(truth_path, index=False)

    return model_path, truth_path, tmp_path / "results" # return the paths to the model and truth CSV files, and a path for results

# Test the helper functions

def test_clean_sample_name_strips_whitespace_and_preserves_missing_values():

    values = pd.Series([" S1 ", "S2", None], dtype="object")

    result = evaluation.clean_sample_name(values)

    assert result.iloc[0] == "S1" # check that whitespace is stripped
    assert result.iloc[1] == "S2" # check that valid sample name is preserved
    assert pd.isna(result.iloc[2]) # check that missing value is preserved
    assert str(result.dtype) == "string" # check that the dtype is string


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (-1, -1),
        (1, 1),
        ("-1", -1),
        ("1.0", 1),
        (np.nan, pd.NA),
        (None, pd.NA),
    ],
)
def test_normalise_label_accepts_valid_labels(value, expected):

    result = evaluation.normalise_label(value)

    if expected is pd.NA:
        assert result is pd.NA # check that missing values are preserved as pd.NA
    else:
        assert result == expected # check that valid labels are correctly normalised to -1 or 1


@pytest.mark.parametrize(
    "value",
    ["anomaly", 0, 2, -2]
)
def test_normalise_label_rejects_unexpected_labels(value):

    with pytest.raises(ValueError, match="expected -1 or 1"): # check that unexpected labels raise a ValueError
        evaluation.normalise_label(value)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (" Fold80 ", "fold80"),
        ("COVERAGE", "coverage"),
        ("", None),
        ("   ", None),
        (None, None),
        (np.nan, None),
    ],
)
def test_clean_feature_standardises_feature_names(value, expected):

    assert evaluation.clean_feature(value) == expected # check that feature names are cleaned and standardised correctly


def test_get_predicted_features_cleans_deduplicates_and_ignores_missing_columns():

    row = pd.Series(
        {
            "shap_1": " Fold80 ",
            "shap_2": "fold80",
            "robust_1": None,
        }
    )

    result = evaluation.get_predicted_features(
        row,
        ["shap_1", "shap_2", "robust_1", "not_present"]
    )

    assert result == {"fold80"} # check predicted features are pulled correctly. 


def test_evaluate_reason_method_returns_true_when_a_feature_matches():

    row = pd.Series(
        {
            "truth_anomaly": -1,
            "truth_features": ["coverage", "fold80"],
            "shap_1": " Fold80 ",
            "shap_2": "insert_size",
        }
    )

    result = evaluation.evaluate_reason_method(
        row, ["shap_1", "shap_2"]
    )

    assert result == {"reason_hit": True} # check that reason_hit is True when a predicted feature matches a truth feature


def test_evaluate_reason_method_returns_false_when_no_feature_matches():

    row = pd.Series(
        {
            "truth_anomaly": -1,
            "truth_features": ["coverage"],
            "shap_1": "fold80",
        }
    )

    result = evaluation.evaluate_reason_method(
        row, ["shap_1"]
    )

    assert result == {"reason_hit": False} # check that reason_hit is False when no predicted feature matches a truth feature


@pytest.mark.parametrize(
    ("truth_anomaly", "truth_features"),
    [
        (1, ["coverage"]),
        (-1, []),
        (-1, None),
        (pd.NA, ["coverage"]),
    ],
)
def test_evaluate_reason_method_returns_na_when_not_evaluable(
    truth_anomaly, truth_features
):

    row = pd.Series(
        {
            "truth_anomaly": truth_anomaly,
            "truth_features": truth_features,
            "shap_1": "coverage",
        }
    )

    result = evaluation.evaluate_reason_method(
        row, ["shap_1"]
    )

    assert pd.isna(result["reason_hit"]) # check that reason_hit is pd.NA when the reason method is wrong

# test evaluate_model()
# check that the evaluation function returns expected metrics and writes expected CSV files
# check that TP, TN, FP, FN are correctly identified and counted
# check that SHAP and robust Z explanations are only present for predicted outliers
def test_evaluate_model_returns_metrics_and_writes_expected_csvs(
    evaluation_files,
):

    model_path, truth_path, out_dir = evaluation_files

    result = evaluation.evaluate_model(
        model_path,
        truth_path,
        assay="st",
        version="v2_v3",
        out_dir=out_dir,
    )

    # Four samples are shared.
    assert len(result["matched"]) == 4

    # One truth sample and one model sample are unmatched.
    assert len(result["evaluation"]) == 5

    metrics = result["classification_metrics"]

    assert metrics["matched_samples"] == 4
    assert metrics["missing_truth_samples"] == 1
    assert metrics["model_samples_not_in_truth"] == 1

    # S1 = TN, S2 = TP, S3 = FN, S4 = FP.
    assert result["confusion_matrix"].tolist() == [
        [1, 1],
        [1, 1],
    ]

    assert metrics["true_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
    assert metrics["true_positives"] == 1

    # Only predicted outliers should have explanation features.
    normal_samples = result["matched"].loc[
        result["matched"]["anomaly_label"] == 1
    ]

    for column in [
        "shap_1",
        "shap_2",
        "robust_z_1",
        "robust_z_2",
    ]:
        assert normal_samples[column].isna().all()

    assert set(result["reason_summary"]["method"]) == {
        "shap",
        "robust_z",
    }

    expected_files = {
        "sample_level_evaluation.csv",
        "missing_truth_samples.csv",
        "model_samples_not_in_truth.csv",
        "classification_metrics.csv",
        "reason_evaluation_by_sample.csv",
        "reason_method_comparison.csv",
    }

    actual_files = {
        path.name for path in result["result_dir"].iterdir()
    }

    assert expected_files.issubset(actual_files)


def test_evaluate_model_normalises_sample_names_and_string_labels(
    evaluation_files,
):

    model_path, truth_path, out_dir = evaluation_files

    result = evaluation.evaluate_model(
        model_path, truth_path, "st", "v2", out_dir
    )

    assert "S1" in result["matched"]["sample_name"].tolist()

    assert result["matched"]["anomaly_label"].dtype.kind in "iu"


def test_evaluate_model_rejects_missing_model_columns(
    evaluation_files,
):

    model_path, truth_path, out_dir = evaluation_files

    model = pd.read_csv(model_path).drop(columns=["shap_1"])
    model.to_csv(model_path, index=False)

    with pytest.raises(ValueError, match="Missing model columns"):

        evaluation.evaluate_model(
            model_path, truth_path, "st", "v2", out_dir
        )

def test_evaluate_model_rejects_missing_truth_columns(
    evaluation_files,
):

    model_path, truth_path, out_dir = evaluation_files

    truth = pd.read_csv(truth_path).drop(columns=["anomaly_label"])
    truth.to_csv(truth_path, index=False)

    with pytest.raises(ValueError, match="Missing truth-set columns"):

        evaluation.evaluate_model(
            model_path, truth_path, "st", "v2", out_dir
        )

def test_evaluate_model_rejects_truth_without_truth_feature_columns(
    evaluation_files,
):

    model_path, truth_path, out_dir = evaluation_files

    truth = pd.read_csv(truth_path).drop(
        columns=["truth_feature_1", "truth_feature_2"]
    )

    truth.to_csv(truth_path, index=False)

    with pytest.raises(ValueError, match="No truth_feature_"):

        evaluation.evaluate_model(
            model_path, truth_path, "st", "v2", out_dir
        )

@pytest.mark.parametrize(
    "duplicate_in",
    ["model", "truth"]
)
def test_evaluate_model_rejects_duplicate_sample_names(
    evaluation_files, duplicate_in
):

    model_path, truth_path, out_dir = evaluation_files

    if duplicate_in == "model":

        data = pd.read_csv(model_path)
        data.loc[len(data)] = data.iloc[0]
        data.to_csv(model_path, index=False)

        message = "Duplicate sample_name values in model output"

    else:

        data = pd.read_csv(truth_path)
        data.loc[len(data)] = data.iloc[0]
        data.to_csv(truth_path, index=False)

        message = "Duplicate sample_name values in truth set"

    with pytest.raises(ValueError, match=message):

        evaluation.evaluate_model(
            model_path, truth_path, "st", "v2", out_dir
        )


def test_evaluate_model_rejects_invalid_anomaly_label(
    evaluation_files,
):

    model_path, truth_path, out_dir = evaluation_files

    model = pd.read_csv(model_path)
    model.loc[0, "anomaly_label"] = 0
    model.to_csv(model_path, index=False)

    with pytest.raises(ValueError, match="Unexpected anomaly label"):

        evaluation.evaluate_model(
            model_path, truth_path, "st", "v2", out_dir
        )


def test_evaluate_model_rejects_no_matching_sample_names(
    evaluation_files,
):

    model_path, truth_path, out_dir = evaluation_files

    model = pd.read_csv(model_path)
    model["sample_name"] = [
        f"MODEL_{i}" for i in range(len(model))
    ]

    model.to_csv(model_path, index=False)

    with pytest.raises(ValueError, match="No sample names matched"):

        evaluation.evaluate_model(
            model_path, truth_path, "st", "v2", out_dir
        )


def test_evaluate_model_rejects_missing_predictions_for_matched_samples(
    evaluation_files,
):

    model_path, truth_path, out_dir = evaluation_files

    model = pd.read_csv(model_path)

    model.loc[
        model["sample_name"] == "S1",
        "anomaly_label"
    ] = np.nan

    model.to_csv(model_path, index=False)

    with pytest.raises(ValueError, match="missing anomaly_label"):

        evaluation.evaluate_model(
            model_path, truth_path, "st", "v2", out_dir
        )


def test_evaluate_model_handles_null_sample_names(
    evaluation_files,
):

    model_path, truth_path, out_dir = evaluation_files

    model = pd.read_csv(model_path)
    truth = pd.read_csv(truth_path)

    model.loc[len(model)] = model.iloc[0]
    model.loc[len(model) - 1, "sample_name"] = np.nan

    truth.loc[len(truth)] = truth.iloc[0]
    truth.loc[len(truth) - 1, "sample_name"] = np.nan

    model.to_csv(model_path, index=False)
    truth.to_csv(truth_path, index=False)

    result = evaluation.evaluate_model(
        model_path, truth_path, "st", "v2", out_dir
    )

    assert result["evaluation"]["sample_name"].notna().all()


# Test the graph functions 
def _minimal_graph_results(
    include_scores=True,
    include_reasons=True,
):

    matched = pd.DataFrame(
        {
            "truth_anomaly": [1, -1, -1, 1],
            "anomaly_label": [1, -1, 1, -1],

            # Explanations only for predicted outliers (S2 and S4).
            "shap_1": [None, "fold80", None, "coverage"],
            "shap_2": [None, "insert_size", None, None],

            "robust_z_1": [None, "insert_size", None, "coverage"],
            "robust_z_2": [None, None, None, None],

            "truth_features": [
                [],
                ["fold80"],
                ["coverage"],
                ["insert_size"],
            ],
        }
    )

    if include_scores:
        matched["anomaly_score"] = [
            0.1, -0.3, -0.2, -0.1
        ]

    reason_results = matched.copy()

    reason_summary = pd.DataFrame(
        {
            "method": ["shap", "robust_z"],
            "reason_hit_rate": [0.5, 1.0],
        }
    )

    if not include_reasons:
        reason_results["truth_features"] = [
            [], [], [], []
        ]

    return {
        "confusion_matrix": np.array([
            [1, 1],
            [1, 1],
        ]),

        "classification_metrics": {
            "anomaly_precision": 0.5,
            "anomaly_recall_sensitivity": 0.5,
            "specificity": 0.5,
            "anomaly_f1": 0.5,
            "balanced_accuracy": 0.5,
        },

        "y_true": pd.Series([1, -1, -1, 1]),
        "y_pred": pd.Series([1, -1, 1, -1]),

        "matched": matched,
        "reason_results": reason_results,
        "reason_summary": reason_summary,
    }


def test_graphs_for_evaluation_saves_expected_plots(
    tmp_path, configured_columns
):

    results = _minimal_graph_results()

    evaluation.graphs_for_evaluation(
        results,
        "st",
        "v2",
        tmp_path,
    )

    expected_plots = {
        "confusion_matrix.png",
        "classification_metrics.png",
        "reason_hit_rate_comparison.png",
        "anomaly_score_distributions.png",
        "reason_feature_frequencies.png",
    }

    actual_plots = {
        path.name for path in tmp_path.iterdir()
    }

    assert expected_plots.issubset(actual_plots)


def test_graphs_for_evaluation_skips_score_plot_when_scores_absent(
    tmp_path, configured_columns, capsys
):

    results = _minimal_graph_results(
        include_scores=False
    )

    evaluation.graphs_for_evaluation(
        results,
        "haem",
        "v3",
        tmp_path,
    )

    assert not (
        tmp_path / "anomaly_score_distributions.png"
    ).exists()

    assert "No 'anomaly_score' column" in capsys.readouterr().out


def test_graphs_for_evaluation_skips_feature_plot_when_no_truth_reasons(
    tmp_path, configured_columns, capsys
):

    results = _minimal_graph_results(
        include_reasons=False
    )

    evaluation.graphs_for_evaluation(
        results,
        "haem",
        "v3",
        tmp_path,
    )

    assert not (
        tmp_path / "reason_feature_frequencies.png"
    ).exists()

    assert "No reference reasons available" in capsys.readouterr().out


def test_graphs_for_evaluation_handles_missing_anomaly_scores(
    tmp_path, configured_columns
):

    results = _minimal_graph_results(
        include_scores=True
    )

    results["matched"]["anomaly_score"] = [
        np.nan, np.nan, np.nan, np.nan
    ]

    evaluation.graphs_for_evaluation(
        results,
        "st",
        "v3",
        tmp_path,
    )

    # The column exists, but there are no usable scores to plot.
    assert not (
        tmp_path / "anomaly_score_distributions.png"
    ).exists()

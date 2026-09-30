from pipeline.preprocess import run_preprocessing
from pipeline.train_individual import run_model_train
from pipeline.evaluate import quality_metric_flag_success, get_truth_set
import config
import os
import pandas as pd
import argparse

def contamination_search(
    summary_qc_metrics: str,
    assay: str,
    version: str,
    out_dir,
    split
):

    X_train, X_test, train_meta, test_meta = run_preprocessing(
        summary_qc_metrics,
        assay,
        out_dir,
        version
    )

    contamination_values = [
        0.01,
        0.03,
        0.05,
        0.08,
        0.10,
        0.15,
        0.20,
        0.25
    ]

    truth_set = get_truth_set("data/raw/", assay)

    all_results = []

    for cont_value in contamination_values:

        print(f"Testing contamination = {cont_value}")

        train_results, test_results = run_model_train(
            X_train.copy(),
            X_test.copy(),
            train_meta,
            test_meta,
            assay,
            version,
            split,
            cont_value,
            os.path.join("models/trained", assay)
        )

        if split == "train":
            scored_explained = train_results

        elif split == "test":
            scored_explained = test_results

        # Find each metric's anomaly-label column
        anomaly_columns = [
            col for col in scored_explained.columns
            if col.endswith("anomaly_label")
        ]

        for anomaly_col in anomaly_columns:

            metric = anomaly_col.replace("_anomaly_label", "")

            print(
                f"  Evaluating metric: {metric}"
            )

            # Keep only the anomaly label for this metric
            metric_results = scored_explained.copy()

            # Make the column expected by quality_metric_flag_success
            metric_results["anomaly_label"] = (
                metric_results[anomaly_col]
            )

            returned_analysis = quality_metric_flag_success(
                metric_results,
                truth_set,
                assay,
                version,
                split
            )

            # Add information about the model
            returned_analysis["metric"] = metric
            returned_analysis["contamination"] = cont_value

            all_results.append(returned_analysis)

    results_df = pd.DataFrame(all_results)

    os.makedirs("outputs/evaluation", exist_ok=True)

    results_df.to_csv(
        os.path.join(
            "outputs/evaluation",
            f"{assay}_{version}_{split}_contamination_search.csv"
        ),
        index=False
    )

    return results_df

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description = "Evaluate a model against known flagged fails"
    )

    parser.add_argument(
        "--assay",
        choices=["st", "haem"],
        help="Assay type: st or haem"
    )

    parser.add_argument(
        "--version",
        choices=["v2", "v3", "v2_v3"],
        help="Cature version of pipeline: v2, v3 or v2_v3"
    )

    parser.add_argument(
            "--split",
            choices=["test", "train" ],
            required=False,
            help="Cature version of pipeline: v2, v3 or v2_v3"
        )



    args= parser.parse_args()

    contamination_search(config.SUMMARY_QC_METRICS, assay=args.assay, version=args.version, out_dir=None, split=args.split)
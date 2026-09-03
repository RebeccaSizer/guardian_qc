from pipeline.preprocess import run_preprocessing
from pipeline.train import run_model_train
from pipeline.evaluate import quality_metric_flag_success, get_truth_set
import config
import os
import pandas as pd
import argparse

def contamination_search(summary_qc_metrics: str, assay: str, version: str, out_dir, split):

    X_train, X_test, train_meta, test_meta = run_preprocessing(summary_qc_metrics, assay, out_dir, version)
    
    contamination = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25]
    all_results = {}

    for cont_value in contamination:
        explained_model_train, explained_model_test = run_model_train(X_train, X_test, train_meta, test_meta, assay, version, cont_value, os.path.join('models/trained', assay ))

        truth_set = get_truth_set('data/raw/', assay)

        if split == 'train':
            scored_explained = explained_model_train.copy()
        elif split == 'test':
            scored_explained = explained_model_test.copy()

        returned_analysis = quality_metric_flag_success(scored_explained, truth_set, assay, version, split)

        all_results[cont_value] = returned_analysis
    
        results_df = pd.DataFrame.from_dict(
        all_results,
        orient="index"
        )
    
        results_df.to_csv(os.path.join("outputs/evaluation/", f"{args.assay}_{args.version}_{args.split}.csv"))

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description = "Evaluate a model against known flagged fails"
    )

    parser.add_argument(
        "--assay",
        choices=["ST", "haem"],
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
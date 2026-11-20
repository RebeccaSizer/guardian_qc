import argparse
import os
import pandas as pd
import config

from pipeline.preprocess import run_preprocessing
from pipeline.train_composite import run_model_train
from pipeline.evaluate import evaluate_model, graphs_for_evaluation
from utils.logger import logging


parser = argparse.ArgumentParser(
    description="Run the Guardian-QC Isolation Forest pipeline."
)

parser.add_argument(
    "--assay",
    choices=["st", "haem"],
    required=True,
    help='Assay to analyse: "st" or "haem"',
)

parser.add_argument(
    "--version",
    choices=["v2", "v3", "v2_v3"],
    required=True,
    help="Capture version: v2, v3 or v2_v3",
)

parser.add_argument(
    "--split",
    choices=["train", "test"],
    required=False,
    help='Split used for evaluation: Train or Test"',
)

args = parser.parse_args()

X_train, X_test, train_meta, test_meta = run_preprocessing(
        file_path=os.path.join('data', config.SUMMARY_QC_METRICS),
        assay=args.assay,
        version=args.version,
        out_dir_file='data',
        out_dir_model=os.path.join(
            config.PREPROCESSING_OUTDIR,
            args.assay
        ),
    )

if args.assay == 'haem':
    contamination = config.HAEM_CONCENTRATION
    n_estimators = config.HAEM_N_ESTIMATORS
    max_samples = config.HAEM_MAX_SAMPLES
    max_features = config.HAEM_MAX_FEATURES
    random_state = config.HAEM_RANDOM_STATE
    
elif args.assay == 'st':
    contamination = config.ST_CONCENTRATION
    n_estimators = config.ST_N_ESTIMATORS
    max_samples = config.ST_MAX_SAMPLES
    max_features = config.ST_MAX_FEATURES
    random_state = config.ST_RANDOM_STATE

# The default values are: contamination 0.08 for haem and 15% for st. n_estimators = 100, max_samples = 256, max_features = 1.0
explained_model_train, explained_model_test = run_model_train(X_train, X_test, train_meta, test_meta, args.assay, args.version, contamination, n_estimators, max_samples, max_features, os.path.join(config.TRAINED_OUTPUT_DATA, args.assay, args.version ))


results = evaluate_model(
    f'data/trained/{args.assay}/{args.version}/{args.assay}_{args.version}_c{contamination}_n{n_estimators}_ms{max_samples}_mf{max_features}_test_scored.csv',
    truth_file=os.path.join(config.TRUTH_SET_DIR, f'truth_set_{args.assay}_500.csv'),
    assay=args.assay,
    version=args.version,
    out_dir=config.EVALUATE_OUTDIR,
)

graphs_for_evaluation(results, assay=args.assay, version=args.version, out_dir=os.path.join("outputs/graphs/evaluate/", args.assay, args.version))
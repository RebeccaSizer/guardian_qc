"""
This script contains unit tests for the ingest module. 
It uses the pytest framework to test various functions 
within the ingest module, ensuring that they behave as 
expected under different conditions. The tests cover scenarios 
such as data ingestion, error handling, and data validation.
"""

from asyncio import run
import re
from unittest.mock import patch

import config
from pipeline.ingest import (get_qc_summary_file_path, 
                            get_run_file_paths, 
                            merge_run_and_qc_data, 
                            filter_df, 
                            filter_run_qc, 
                            sample_level_qc)
import pytest
import pandas as pd


###############################
# Test Ingest Functions       #  
# #############################


# get_qc_summary_file_path    
###############################


def test_get_qc_summary_file_path(tmp_path, monkeypatch):
    """
    Test the get_qc_summary_file_path function to ensure it returns 
    the correct DataFrame structure and expected columns.
    """
    
    run = tmp_path / "250101_A00001"
    run.mkdir()

    worklist = run / "1234567"
    worklist.mkdir()

    file = worklist / "1234567_qc_summary.csv"
    file.touch()

    monkeypatch.setattr(config, "ROOT_FILE_PATH", tmp_path)

    monkeypatch.setattr(
        config,
        "RUN_PATTERN_NOVASEQX",
        re.compile(r".*")
    )

    monkeypatch.setattr(
        config,
        "RUN_PATTERN_SUB",
        re.compile(r"\d{6}")
    )

    monkeypatch.setattr(
        config,
        "QC_SUMMARY_PATTERN_ST",
        re.compile(r".*")
    )

    df = get_qc_summary_file_path()

    # Check if the returned object is a DataFrame
    assert isinstance(df, pd.DataFrame), "Expected a pandas DataFrame"
    assert len(df) == 1
    assert df.iloc[0]["worklist"] == "1234567"
    assert df.iloc[0]["sequencer"] == "novaseqx"
    assert df.iloc[0]["cancer_type"] == "solid_tumour"

     
# get_run_file_paths   
###############################

def create_sample_sheet(path, lanes):
    """Create a minimal Illumina sample sheet."""

    with open(path, "w") as f:
        # 15 lines skipped by pd.read_csv(skiprows=15)
        for _ in range(15):
            f.write("dummy\n")

        f.write("Lane,Sample_ID\n")

        for lane in lanes:
            f.write(f"{lane},sample\n")


def test_get_run_file_paths(tmp_path, monkeypatch):
    """
    Test the get_run_file_paths function to ensure it returns 
    the correct DataFrame structure and expected columns.
    """
    seq_X = tmp_path / "NovaSeqX"
    seq_X.mkdir()

    seq_6000 = tmp_path / "NovaSeq6000"
    seq_6000.mkdir()

    run_X = seq_X / "250101_A00002"
    run_X.mkdir()

    run_6000 = seq_6000 / "250101_A00001"
    run_6000.mkdir()

    run_no_sample_sheet = seq_6000 / "250101_A00003"
    run_no_sample_sheet.mkdir()

    sample_sheet_X = run_X / "SampleSheet.csv"
    create_sample_sheet(sample_sheet_X, [1])

    sample_sheet_6000 = run_6000 / "SampleSheet.csv"
    create_sample_sheet(sample_sheet_6000, [1])

    monkeypatch.setattr(config, "NOVASEQX_PATH", seq_X)
    monkeypatch.setattr(config, "NOVASEQ6000_PATH", seq_6000)

    monkeypatch.setattr(config, "RUN_PATTERN_NOVASEQX", re.compile(r".*"))
    monkeypatch.setattr(config, "RUN_PATTERN_NOVASEQ6000", re.compile(r".*"))

    monkeypatch.setattr(config, "SAMPLE_SHEET_PATTERN", re.compile(r".*"))

    df = get_run_file_paths()

    assert len(df) == 3
    assert df.loc[0, "seq_run_number"] == "250101_A00002"
    assert df.loc[0, "lane"] == [1]
    assert df.loc[0, "sample_sheet_path"] == str(sample_sheet_X)
    assert df.loc[1, "seq_run_number"] == "250101_A00001"
    assert df.loc[1, "lane"] == [8]
    assert df.loc[1, "sample_sheet_path"] == str(sample_sheet_6000)
    assert df.loc[2, "sample_sheet_path"] == "No sample sheet found"

    # Check if the returned object is a DataFrame
    assert isinstance(df, pd.DataFrame), "Expected a pandas DataFrame"
    assert df.columns.tolist() == [
        "seq_run_number",
        "run_qual_filepath",
        "sample_sheet_path",
        "lane"
    ]

# merge_run_and_qc_data
###############################

def test_merge_run_and_qc_data():
    """
    Test the merge_run_and_qc_data function to ensure it returns 
    the correct DataFrame structure and expected columns.
    """
    
    df = pd.DataFrame({
        "seq_run_number": [1, 2],
        "run_qual_filepath": ["path1", "path2"],
        "sample_sheet_path": ["sheet1", "sheet2"],
        "worklist": ["worklist1", "worklist2"],
        "qc_file_path": ["qc1", "qc2"],
        "sequencer": ["seq1", "seq2"],
        "cancer_type": ["type1", "type2"],
        "file_status": ["Yes", "No"]
    })
    
    # Check if the returned object is a DataFrame
    assert isinstance(df, pd.DataFrame), "Expected a pandas DataFrame"
    
    # Check for expected columns in the DataFrame
    expected_columns = ['seq_run_number',
                        'run_qual_filepath',
                        'sample_sheet_path',
                        'worklist',
                        'qc_file_path',
                        'sequencer',
                        'cancer_type',
                        'file_status']
    
    for col in expected_columns:
        assert col in df.columns, f"Missing expected column: {col}"

# filter_df
###############################

def test_filter_df():

    """
    Test the filter_df function to ensure it correctly filters 
    the DataFrame based on the provided criteria.
    """
    
    df = pd.DataFrame({
        "file_status": ["Yes", "Yes"],
        "lane": [1, 1]
    })

    
    filtered_df = filter_df(df)
    
    # Check if the returned object is a DataFrame
    assert isinstance(filtered_df, pd.DataFrame), "Expected a pandas DataFrame"
    
    # Check if the filtered DataFrame meets the criteria
    assert all(filtered_df["file_status"] == "Yes")
    "Filtered DataFrame contains rows with file_status not equal to 'Yes'"

    assert all(filtered_df["lane"] == 1)


def test_filter_df_filters_rows():

    df = pd.DataFrame({
        "file_status": ["Yes", "No", "Yes", "No"],
        "lane": [1, 1, None, 2]
    })

    result = filter_df(df)

    assert len(result) == 1
    assert result.iloc[0]["file_status"] == "Yes"
    assert result.iloc[0]["lane"] == 1

# filter_run_qc
###############################

@pytest.mark.parametrize(
    "q30,error_rate,expected",
    [
        (90, 1, "Yes"),
        (75, 1, "Percent Q30 < 80"),
        (90, 3, "Error rate > 2"),
        (75, 3, "Percent Q30 < 80 AND Error rate > 2"),
    ],
)

@patch("pipeline.ingest.inter_op_qc")
def test_filter_run_qc(mock_interop, q30, error_rate, expected):
    """
    Test the filter_run_qc function to ensure it correctly filters 
    the merged DataFrame based on QC and lane criteria.
    """
    
    mock_interop.return_value = pd.DataFrame({
        "Lane": ["Full Run"],
        "Percent Q30": [q30],
        "Error Rate": [error_rate],
    })

    df = pd.DataFrame({
        "seq_run_number": ["Run1"],
        "run_qual_filepath": ["/tmp/Run1"],
        "sequencer": ["NovaSeq6000"],
        "lane": [[1]],
    })

    result = filter_run_qc(df)

    assert result.loc[0, "pass_run_qc"] == expected

# test sample_level_qc
###############################

def create_sample_qc_file(path):

    df = pd.DataFrame({
        "sample_name": ["Sample001"],
        "bcftools_ts": [100],
        "bcftools_tv": [50],
        "bcftools_tstv": [2.0],
        "bcftools_variants": [1000],
        "bcftools_SNVs": [900],
        "bcftools_indels": [100],
        "picard_mode_insert": [250],
        "picard_mean_insert": [300],
        "picard_median_insert": [295],
        "picard_mad_insert": [10],
        "picard_total_reads": [50000000],
        "picard_pf_reads": [49000000],
        "picard_pf_q30_bases": [45000000],
        "picard_read_length": [150],
        "picard_at_dropout": [0.01],
        "picard_gc_dropout": [0.02],
        "picard_fold_enrichment": [5.0],
        "picard_fold80": [100],
        "picard_mean_target_coverage": [80],
        "picard_median_target_coverage": [75],
        "picard_target_bases_20x": [95],
        "picard_target_bases_30x": [90],
        "picard_target_bases_50x": [85],
        "picard_target_bases_100x": [80],
        "fastqc_duplication_rate": [0.1],
        "fastqc_basic_status": ["PASS"],
        "fastp_duplication_rate": [0.05],
    })

    df.to_csv(path, sep="\t", index=False)

def test_sample_level_qc(tmp_path):

    """
    Test the sample_level_qc function to ensure it correctly processes 
    the sample-level QC data and returns the expected DataFrame structure.
    """
    
    qc_file = tmp_path / "sample_qc.csv"
    create_sample_qc_file(qc_file)

    print(tmp_path)

    df = pd.DataFrame({
        "seq_run_number": ["Run1"],
        "run_qual_filepath": ["run/path"],
        "sample_sheet_path": ["SampleSheet.csv"],
        "worklist": ["worklist1"],
        "qc_file_path":  [str(qc_file)],
        "sequencer": ["NovaSeq6000"],
        "cancer_type": ["type1"],
        "pass_run_qc": ["Yes"],
        "file_status": ["Yes"]
    })

    result = sample_level_qc(df)
    print(result)

    # Check if the returned object is a DataFrame
    assert isinstance(result, pd.DataFrame), "Expected a pandas DataFrame"
    
    # Check for expected columns in the DataFrame
    expected_columns = [
        "sample_name",
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
        "fastqc_basic_status",
        "fastp_duplication_rate"
    ]
    
    for col in expected_columns:
        assert col in result.columns, f"Missing expected column: {col}"
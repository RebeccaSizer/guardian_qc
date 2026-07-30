from tools.modules.ml_data_preprocessing import sample_level_qc
from tools.utils.utils_cancer import get_run_file_paths, get_qc_summary_file_path, merge_run_and_qc_data, filter_df
from tools.utils.logger import logger
from tools.modules.filter_by_run_qc import filter_run_qc



output_run_folder = get_run_file_paths()
output_qc_file = get_qc_summary_file_path()
merged_output = merge_run_and_qc_data(output_run_folder, output_qc_file)
filtered_df = filter_df(merged_output)
filtered_output = filter_run_qc(filtered_df)
summary_qc_metrics_df = sample_level_qc(filtered_output)

summary_qc_metrics_df.to_csv("outputs/summary_qc_metrics.csv", sep = "\t", index = False)
print(summary_qc_metrics_df)
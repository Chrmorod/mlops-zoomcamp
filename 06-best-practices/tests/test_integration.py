import os
import sys
import pickle
import pandas as pd
import pyarrow
import fastparquet
import batch_s3




def test_integration():
    os.environ["S3_ENDPOINT_URL"] = "http://localhost:4566"
    os.system("python batch_s3.py 2023 1")

    output_file = "s3://nyc-duration/out/2023-01.parquet"

    df_result = batch_s3.read_data(output_file)
    total_predicted_duration = df_result['predicted_duration'].sum()

    print("Sum of predicted durations:", total_predicted_duration)

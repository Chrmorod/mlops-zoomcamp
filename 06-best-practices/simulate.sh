#!/bin/sh

export INPUT_FILE_PATTERN="https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet"
#export INPUT_FILE_PATTERN="s3://nyc-duration/{year:04d}-{month:02d}.parquet"
export OUTPUT_FILE_PATTERN="s3://nyc-duration/out/{year:04d}-{month:02d}.parquet"
export S3_ENDPOINT_URL="http://localhost:4566"

python batch_s3.py 2023 1
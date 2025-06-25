import pandas as pd
import os

def create_test_dataframe():
    data = {
        "tpep_pickup_datetime": pd.to_datetime([
            "2023-01-01 00:15:00",
            "2023-01-01 00:30:00"
        ]),
        "tpep_dropoff_datetime": pd.to_datetime([
            "2023-01-01 00:25:00",
            "2023-01-01 00:50:00"
        ]),
        "PULocationID": [10, 20],
        "DOLocationID": [30, 40]
    }
    df_input = pd.DataFrame(data)
    return df_input

def main(year, month):
    s3_endpoint_url = os.getenv("S3_ENDPOINT_URL", "http://localhost:4566")
    bucket = "nyc-duration"
    filename = f"{year:04d}-{month:02d}.parquet"
    input_file = f"s3://{bucket}/{filename}"

    options = {
        "client_kwargs": {
            "endpoint_url": s3_endpoint_url,
        }
    }

    df_input = create_test_dataframe()

    df_input.to_parquet(
        input_file,
        engine='pyarrow',
        compression=None,
        index=False,
        storage_options=options
    )

    print(f"Archivo parquet guardado en {input_file}")

if __name__ == "__main__":
    import sys

    year = int(sys.argv[1])
    month = int(sys.argv[2])
    main(year, month)

#!/usr/bin/env python
# coding: utf-8

# In[26]:


get_ipython().system('pip freeze | grep scikit-learn')


# In[27]:


get_ipython().system('python -V')


# In[28]:


import pickle
import pandas as pd
import numpy as np


# In[29]:


with open('model.bin', 'rb') as f_in:
    dv, model = pickle.load(f_in)


# In[30]:


categorical = ['PULocationID', 'DOLocationID']

def read_data(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')

    return df


# In[31]:


df = read_data('https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-03.parquet')


# In[32]:


dicts = df[categorical].to_dict(orient='records')
X_val = dv.transform(dicts)
y_pred = model.predict(X_val)


# In[33]:


np.std(y_pred)#np.float64(6.247488852238703)


# In[34]:


year = 2023
month = 3
df_result = pd.DataFrame({
    'ride_id': f'{year:04d}/{month:02d}_' + df.index.astype('str'),
    'prediction': y_pred
})


# In[35]:


df_result.to_parquet(
    './predictions.parquet',
    engine='pyarrow',
    compression=None,
    index=False
)


# In[49]:


import pickle
import pandas as pd

categorical = ['PULocationID', 'DOLocationID']

def pipeline(model_path: str, year: int, month: int) -> float:

    with open(model_path, 'rb') as f_in:
        dv, model = pickle.load(f_in)

    df = pd.read_parquet(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet')

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    df_result = pd.DataFrame({
        'ride_id': df['ride_id'],
        'prediction': y_pred
    })

    output_file = f"predictions/result-{year}-{month}.parquet"
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )

    print(f"Saved predictions to {output_file}")
    return y_pred.mean()

if __name__ == "__main__":
    pipeline("model.bin", 2023, 3)



# In[51]:


import os
print(os.path.getsize('./predictions/result-2023-3.parquet') / 1024**2, "MB")


# In[ ]:





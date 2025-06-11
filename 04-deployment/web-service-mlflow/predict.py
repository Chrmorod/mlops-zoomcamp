import pickle
import mlflow
from flask import Flask, request, jsonify
from waitress import serve 
from mlflow.tracking import MlflowClient

RUN_ID = '053d9955cf37469b97ef2b8f3db6a419'
MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
artifact_path = "dict_vectorizer.bin"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("green-taxi-duration")

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)


path = client.download_artifacts(run_id=RUN_ID, path=artifact_path)
print(f'downloading the dict vectorizer to {path}')

with open(path, 'rb') as f_in:
    (dv, model) = pickle.load(f_in)

logged_model = f'runs:/{RUN_ID}/model'
model = mlflow.pyfunc.load_model(logged_model)



def prepare_features(ride):
    features = {}
    features['PU_DO'] = '%s_%s' % (ride['PULocationID'] , ride['DOLocationID'])
    features['trip_distance'] = ride['trip_distance']
    return features

def predict(features):
    X = dv.transform(features)
    preds = model.predict(X)
    return preds[0]

app = Flask('duration-prediction')

@app.route('/predict',methods=['POST'])
def predict_endpoint():
    ride = request.get_json()
    features = prepare_features(ride)
    pred = predict(features)

    result = {
        'duration': pred,
        'model_version': RUN_ID
    }
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug = True, host = '0.0.0.0', port=9696)
    #serve(app, host="0.0.0.0", port=9696)
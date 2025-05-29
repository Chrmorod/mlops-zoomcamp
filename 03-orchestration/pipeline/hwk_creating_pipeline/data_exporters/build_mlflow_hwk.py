import mlflow
import mlflow.sklearn

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter




@data_exporter
def export_model(inputs, **kwargs):

    dv, model = inputs
    mlflow.set_tracking_uri("file:./mlruns")
    ##Declare my name of the experiment
    experiment_name = "mlops_hwk"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        #Create the experiment if not exist
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    #set this experiment name
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run():

        mlflow.sklearn.log_model(model, artifact_path="model") # we save model

        mlflow.log_param("num_features", len(dv.feature_names_))

        run_id = mlflow.active_run().info.run_id
        print(f"Experiment {experiment_id}, Model saved in MLflow with Run ID: {run_id}")
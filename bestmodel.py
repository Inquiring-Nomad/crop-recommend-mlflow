import os.path

import mlflow
from pprint import pprint
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import pickle

if __name__ == "__main__":
    print("Running Best Model entrypoint")
    productionPath = 'production-app'

  #Search for the run with the best score
    run = MlflowClient().search_runs(
        experiment_ids=["1"],
        filter_string="",
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.test_score DESC"]
    )[0]
    run_id = run.info.run_id
    artifacts_uri = run.info.artifact_uri

    source_name = run.data.tags['mlflow.source.name']
    client = MlflowClient()
    #Runs tagged as production
    #Search for the model versions of the run
    for mv in client.search_model_versions(f"run_id='{run_id}'"):
        mv = dict(mv)
        logged_model = mv["source"]
        current_version = mv["version"]
        model_name = mv["name"]
        current_stage = mv['current_stage']
        # if (current_stage == 'Production'):
        #     print("No change in production model")
        #     break

        model_path = os.path.join(source_name, logged_model)
        # artifacts_path = os.path.join(source_name,artifacts_uri)
        # scaler = client.download_artifacts()
        local_path_scaler = client.download_artifacts(run.info.run_id, "scaler", productionPath)
        local_path_encoder = client.download_artifacts(run.info.run_id, "labelencoder", productionPath)
        print("Artifacts downloaded in: {}".format(local_path_scaler))
        print("Artifacts: {}".format(os.listdir(local_path_scaler)))
        print("Artifacts downloaded in: {}".format(local_path_encoder))
        print("Artifacts: {}".format(os.listdir(local_path_encoder)))

        loaded_model = mlflow.sklearn.load_model(model_path)
        #Transition to 'Production' stage
        client.transition_model_version_stage(
            name=model_name,
            version=current_version,
            stage="Production"
        )
        with mlflow.start_run(run_id=run_id) as pr:
            mlflow.set_tag("production.current",True)
            mlflow.set_tag("production.name", model_name)
            mlflow.set_tag("production.version", current_version)

        #Switch previous production models back to None
        filter_string = "tags.production.current = 'True'"
        runs = client.search_runs(["1"], run_view_type=ViewType.ACTIVE_ONLY,
                                  filter_string=filter_string)
        for r in runs:
            old_run_id = r.info.run_id

            tags = {k: v for k, v in r.data.tags.items() if not k.startswith("mlflow.")}
            prod_name =  tags['production.name']
            prod_version=  tags['production.version']
            if run_id != old_run_id:
                client.transition_model_version_stage(
                    name=prod_name,
                    version=prod_version,
                    stage="None"
                )
                with mlflow.start_run(run_id=old_run_id) as pr:
                    mlflow.set_tag("production.current", False)





        pickle.dump(loaded_model, open(os.path.join(productionPath, 'model.pkl'), 'wb'))

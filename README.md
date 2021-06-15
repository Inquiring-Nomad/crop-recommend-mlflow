#Crop recommendation

Using a kaggle dataset taken from :

https://www.kaggle.com/atharvaingle/crop-recommendation-dataset

### MLFLOW

The lifecycle of the project as well as the tracking is done with [MLFLOW](https://mlflow.org/)

MLflow is an open source platform to manage the ML lifecycle, including experimentation, reproducibility, deployment, and a central model registry.

### Models:

Three multi-class classification algorithms are used:

- XGB Classifier (with Gridsearch CV) 
- KNN
- Random Forest (with Gridsearch CV) 

### The implementation of MLFLOW

Tracking of the metrics is done on an mlflow tracking server , initiated with:

`mlflow server 
--backend-store-uri sqlite:///mlflow.db 
--default-artifact-root ./artifacts 
--host 127.0.0.1 -p 1234 `

The artifacts are generated in ./artifacts folder and the metrics are logged in sqlite:///mlflow.db 

These three commands train the classifier and log the artifacts and metrics to MLFLow. 

They create an experiment named _"Crop prediction"_ and use three different entry points as defined in the **MLProject** file

The model with the best test score is copied over to the `production-app/` folder

`mlflow run --no-conda --experiment-name 'Crop prediction' -e random-forest   .`

`mlflow run --no-conda --experiment-name 'Crop prediction' -e knn   .`

`mlflow run --no-conda --experiment-name 'Crop prediction' -e xgb   .`

### Production app

The UI of the app is created with [streamlit](https://streamlit.io/).

It is deployed as a heroku app , available at :

###Data analysis

Exploratory data analysis is done with the help of Autoviz and Pandas Profiling. The plots can be found in 

`notebooks/`









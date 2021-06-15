import sys
import pandas as pd
import pickle
import mlflow
from re import match
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, DMatrix


def knn():
    print("KNN")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else '../data/Crop_recommendation.csv'
    algo = sys.argv[2]
    algo()
    exit()
    random_state = sys.argv[2] if len(sys.argv) > 2 else 42
    df = pd.read_csv(path)
    c = df.label.astype('category')
    targets = dict(enumerate(c.cat.categories))
    df['target'] = c.cat.codes

    y = df.target
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # we must apply the scaling to the test set as well that we are computing for the training set

    X_test_scaled = scaler.transform(X_test)



    mlflow.sklearn.autolog()
    param_grid = {
        'n_estimators': [500, 1000, 1500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 6],
        'criterion': ['gini', 'entropy']
    }
    rf = RandomForestClassifier()
    clf = GridSearchCV(rf, param_grid)
    # mlflow.set_experiment('Crop prediction')



    with mlflow.start_run(run_name="Random Forest"):
        mlflow.set_tag('mlflow.runName', "Random Forest")
        clf.fit(X_train_scaled, y_train)
        mlflow.sklearn.eval_and_log_metrics(clf, X_test_scaled, y_test, prefix="test_")
        pickle.dump(scaler, open('scaler.pkl', 'wb'))
        mlflow.log_artifact('scaler.pkl', "scaler")
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="sklearn-model",
            registered_model_name="sk-learn-random-forest-crops",

        )



import sys
import pandas as pd
import os
import pickle
import mlflow
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier, DMatrix

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else 'data/Crop_recommendation.csv'
    random_state = sys.argv[2] if len(sys.argv) > 2 else 42
    df = pd.read_csv(path)
    df['label'] = df['label'].astype('category')
    y = df['label']
    X = df[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    encoder = LabelEncoder()
    y_train_encoded = encoder.fit_transform(y_train)
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
        clf.fit(X_train_scaled, y_train_encoded)

        # Evaluate on test set
        X_test_scaled = scaler.transform(X_test)
        y_test_encoded = encoder.transform(y_test)
        mlflow.sklearn.eval_and_log_metrics(clf, X_test_scaled, y_test_encoded, prefix="test_")
        pickle.dump(scaler, open('scaler.pkl', 'wb'))
        pickle.dump(encoder, open('labelencoder.pkl', 'wb'))
        mlflow.log_artifact('scaler.pkl', "scaler")
        mlflow.log_artifact('labelencoder.pkl', "labelencoder")
        # Delete the file after logging the articats
        print("Deleting files...")
        if os.path.exists("scaler.pkl"):
            os.remove("scaler.pkl")
        if os.path.exists("labelencoder.pkl"):
            os.remove("labelencoder.pkl")
        mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="sklearn-model",
            registered_model_name="sk-learn-random-forest-crops",

        )

    mlflow.run('.', entry_point="best", experiment_name="Crop prediction", use_conda=False)

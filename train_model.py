from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

data = load_iris()

x = data.data
y=data.target

RFC = RandomForestClassifier(n_estimators=100)
RFC.fit(x,y)


joblib.dump(RFC, "iris_classifier_model.pkl")

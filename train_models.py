import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Keep delivery_dataset.csv in the same folder as this script.
df = pd.read_csv("delivery_dataset.csv")

features = df[["distance", "traffic", "weather", "preparation_time"]].copy()
features.columns = ["Distance", "Traffic", "Weather", "Preparation_Time"]

numeric = ["Distance", "Preparation_Time"]
categorical = ["Traffic", "Weather"]

preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical)
])

# Classification: predicts the order_status column.
X_class = features
y_class = df["order_status"].astype(str).str.strip()

classification_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=2
    ))
])
classification_model.fit(X_class, y_class)
joblib.dump(classification_model, "delivery_classification.pkl")

# Regression: predicts actual_delivery_time in minutes.
y_reg = pd.to_numeric(df["actual_delivery_time"], errors="coerce")
valid = y_reg.notna()
regression_model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        min_samples_leaf=2
    ))
])
regression_model.fit(features.loc[valid], y_reg.loc[valid])
joblib.dump(regression_model, "delivery_regression.pkl")

print("Created delivery_classification.pkl and delivery_regression.pkl")

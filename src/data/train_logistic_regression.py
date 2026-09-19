import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# ==============================
# 1. Load training data
# ==============================

df = pd.read_csv("data/processed/training_data.csv")

print("Dataset shape:", df.shape)


# ==============================
# 2. Separate target
# ==============================

target = "blocked_proxy"

X = df.drop(columns=[target])
y = df[target]

# earthquake_id is used ONLY for grouping.
# It must not be used as a model feature.
groups = X["earthquake_id"]

X = X.drop(columns=["earthquake_id", "road_id"])


# ==============================
# 3. Group-based train/test split
# ==============================

gss = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_idx, test_idx = next(
    gss.split(X, y, groups=groups)
)

X_train = X.iloc[train_idx]
X_test = X.iloc[test_idx]

y_train = y.iloc[train_idx]
y_test = y.iloc[test_idx]


print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

print(
    "Unique earthquakes in training:",
    groups.iloc[train_idx].nunique()
)

print(
    "Unique earthquakes in testing:",
    groups.iloc[test_idx].nunique()
)


# ==============================
# 4. Identify columns
# ==============================

categorical_features = [
    "road_type",
    "oneway"
]

numeric_features = [
    "magnitude",
    "earthquake_depth_km",
    "distance_from_epicenter_km",
    "road_length_m",
    "lanes",
    "maxspeed"
]


# ==============================
# 5. Preprocessing
# ==============================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ==============================
# 6. Logistic Regression
# ==============================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=1000,
        random_state=42
    ))
])


# ==============================
# 7. Train
# ==============================

model.fit(X_train, y_train)

print("\nModel trained successfully!")


# ==============================
# 8. Prediction
# ==============================

y_pred = model.predict(X_test)


# ==============================
# 9. Evaluation
# ==============================

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
joblib.dump(
    model,
    "data/processed/logistic_regression_model.pkl"
)

print("\nModel saved successfully!")
# train.py
import pandas as pd
import numpy as np
import joblib, json
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

import matplotlib.pyplot as plt
import seaborn as sns


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("forestfires.csv")

# Set style
sns.set_style("whitegrid")

# 1. Monthly Fire Count
plt.figure(figsize=(8,5))
sns.countplot(x="month", data=df, order=['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec'])
plt.title("Number of Fires per Month")
plt.show()

# 2. Temperature vs Burned Area
plt.figure(figsize=(8,5))
sns.scatterplot(x="temp", y="area", data=df)
plt.title("Temperature vs Burned Area")
plt.show()

# 3. Correlation Heatmap
plt.figure(figsize=(10,6))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# 1) Load
df = pd.read_csv("forestfires.csv")

# 2) Target transform (log1p to reduce skew)
df['area'] = np.log1p(df['area'])

# 3) Features & target
X = df.drop(columns=['area'])
y = df['area']

# 4) categorical & numeric columns
cat_cols = ['month', 'day']
num_cols = [c for c in X.columns if c not in cat_cols]

# 5) Preprocessor + Decision Tree model (shallow tree to keep JSON small)
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols),
], remainder='passthrough')

pipeline = Pipeline([
    ('prep', preprocessor),
    ('dt', DecisionTreeRegressor(random_state=42, max_depth=6))
])

# 6) Split, train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# 7) Eval
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse:.4f}")
print(f"R^2 Score: {r2:.4f}")

# 8) Save whole pipeline for Python demo
joblib.dump(pipeline, "model_pipeline.pkl")
print("Saved model_pipeline.pkl")

# 9) Export decision tree to JSON (so C++ can load it)
# get the trained DecisionTreeRegressor and feature names after preprocessing
prep = pipeline.named_steps['prep']
dt = pipeline.named_steps['dt']

# get onehot encoder feature names
ohe = prep.named_transformers_['cat']
ohe_feature_names = list(ohe.get_feature_names_out(['month', 'day']))
feature_names = ohe_feature_names + num_cols  # order: onehot features then passthrough

# safety check
assert dt.n_features_in_ == len(feature_names), "feature count mismatch"

# helper to turn tree into nested dict
def tree_to_dict(tree, feature_names):
    tree_ = tree.tree_
    def recurse(node):
        if tree_.feature[node] != -2:  # not a leaf
            feat_idx = tree_.feature[node]
            return {
                "feature": feature_names[feat_idx],
                "threshold": float(tree_.threshold[node]),
                "left": recurse(tree_.children_left[node]),
                "right": recurse(tree_.children_right[node])
            }
        else:
            # regression: value is shape (1,1)
            return {"value": float(tree_.value[node][0][0])}
    return recurse(0)

tree_dict = tree_to_dict(dt, feature_names)
with open("tree_model.json", "w") as f:
    json.dump({
        "feature_names": feature_names,
        "tree": tree_dict
    }, f, indent=2)
print("Saved tree_model.json")

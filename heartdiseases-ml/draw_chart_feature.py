# === Draw Feature Importance Chart ===

import kagglehub 
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from kagglehub import KaggleDatasetAdapter

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
# Load the processed training data
dt_train = pd.read_csv("splits/dt_train.csv")

# Separate features and target
X = dt_train.drop("target", axis=1)
y = dt_train["target"]

# Refit a Decision Tree on the selected features (for visualization)
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X, y)

# Get importance values
importances = tree_model.feature_importances_
feature_names = X.columns

# Sort features by importance
feat_imp = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# === Plot ===
plt.figure(figsize=(10,6))
colors = plt.cm.Reds(np.linspace(2, 1, len(feat_imp)))

plt.bar(feat_imp["Feature"], feat_imp["Importance"], color=colors)
plt.title("Feature Importance Scores", fontsize=14)
plt.xlabel("Features", fontsize=12)
plt.ylabel("Feature Importance Score", fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()

# Save or show
plt.savefig("feature_importance_dt.png", dpi=300)
plt.show()

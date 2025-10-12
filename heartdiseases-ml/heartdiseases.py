# Install data

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
# set the path to the file
path = kagglehub.dataset_download("ritwikb3/heart-disease-cleveland")
print("Dataset path:", path)

# Load the correct CSV
df = pd.read_csv(path + "/Heart_disease_cleveland_new.csv")
df = df.drop_duplicates()
print(df.head())
print(df.shape)
# Define column names
COLUMNS = ['age','sex','cp','trestbps','chol','fbs','restecg',
           'thalach','exang','oldpeak','slope','ca','thal','target']
numeric_cols = ['age','trestbps','chol','thalach','oldpeak']
categorical_cols = ['sex','cp','fbs','restecg','exang','slope','ca','thal']

# Apply column names
df.columns = COLUMNS
K_features = 10

# Convert data types
for c in ['age','trestbps','chol','thalach','oldpeak','ca','thal']:
    df[c] = pd.to_numeric(df[c], errors='coerce')

df['target'] = (df['target'] > 0).astype(int)
print("Cleaned shape:", df.shape) 


# Split dataset into train, validation and test sets
TARGET = 'target'
rav_feature_cols = [c for c in df.columns if c != TARGET]
X_all = df[rav_feature_cols]
y_all = df[TARGET]

X_train, X_temp, y_train, y_temp = train_test_split(
    X_all, y_all, test_size=0.2, stratify=y_all, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size = 0.5, stratify=y_temp, random_state=42
)


# A pipeline progress built to automate preprocessing data: handle trash data and normalize data --> then save in csv
# Build a original dataset ( handled data)
cat_proc = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('scaler', MinMaxScaler())
])

num_proc = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocess = ColumnTransformer([
    ('num', num_proc, numeric_cols),
    ('cat', cat_proc, categorical_cols)
])

raw_pipeline = Pipeline([
    ('preprocess', preprocess)
])

X_raw_train = raw_pipeline.fit_transform(X_train, y_train)
X_raw_val  = raw_pipeline.transform(X_val)
X_raw_test = raw_pipeline.transform(X_test)

preprocessed_feature_names = []
for name, transformer, columns in preprocess.transformers_:
    if hasattr(transformer, 'get_feature_names_out'):
        preprocessed_feature_names.extend(transformer.get_feature_names_out(columns))
    else:
        preprocessed_feature_names.extend(columns)


X_raw_train_df = pd.DataFrame(
    X_raw_train, columns=preprocessed_feature_names, index=X_train.index
)
X_raw_val_df = pd.DataFrame(
    X_raw_val, columns=preprocessed_feature_names, index=X_val.index
)
X_raw_test_df = pd.DataFrame(
    X_raw_test, columns=preprocessed_feature_names, index=X_test.index
)

out_dir = Path('splits'); out_dir.mkdir(parents=True, exist_ok=True)
pd.concat([X_raw_train_df, y_train.rename(TARGET)],
         axis=1).to_csv(out_dir/ 'raw_train.csv', index=False)
pd.concat([X_raw_val_df, y_val.rename(TARGET)],
          axis=1).to_csv(out_dir/ 'raw_val.csv', index=False)
pd.concat([X_raw_test_df, y_test.rename(TARGET)],
          axis=1).to_csv(out_dir/ 'raw_test.csv', index=False)


# Create new dataset with Decision Tree

dt_feature_selection_pipeline = Pipeline([
    ('preprocess', preprocess),
    ('decision_tree', DecisionTreeClassifier(random_state=42))
])

dt_feature_selection_pipeline.fit(X_train, y_train)
feature_importance_series = pd.Series(
    dt_feature_selection_pipeline.named_steps['decision_tree'].feature_importances_,
    index=preprocessed_feature_names
)
sorted_feature_importances = feature_importance_series.sort_values(ascending=False)

selected_features = sorted_feature_importances.head(K_features).index.tolist()
print(f'Top {K_features} selected features: {selected_features}')

X_dt_train = X_raw_train_df[selected_features]
X_dt_val =  X_raw_val_df[selected_features]
X_dt_test = X_raw_test_df[selected_features]

pd.concat([X_dt_train, y_train.rename(TARGET)],
          axis=1).to_csv(out_dir / 'dt_train.csv', index=False)
pd.concat([X_dt_val, y_val.rename(TARGET)],
          axis=1).to_csv(out_dir / 'dt_val.csv', index=False)
pd.concat([X_dt_test, y_test.rename(TARGET)],
          axis=1).to_csv(out_dir / 'dt_test.csv', index=False)

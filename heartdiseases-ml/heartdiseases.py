# Install data

import kagglehub 
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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

# Plot the char of important features
plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))

# Create the bar plot using seaborn
# The 'palette="copper"' argument creates the brown-to-orange color gradient
sns.barplot(x=sorted_feature_importances.index, y=sorted_feature_importances.values, palette="Oranges_r")

# Add a title and labels for the axes
plt.title('Feature Importance Scores', fontsize=14)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Feature Importance Score', fontsize=12)

# Display the plot
plt.savefig(f"chart_importance_feature.png", dpi=150, bbox_inches="tight")


# Enrich data by Feature Engineering

def add_new_features_func(df):
    df = df.copy()
    if {'chol', 'age'} <= set(df.columns):
        df['chol_per_age'] = df['chol']/df['age']
    if {'trestbps', 'age'} <= set(df.columns):
        df['bps_per_age'] = df['trestbps']/df['age']
    if {'thalach', 'age'} <= set(df.columns):
        df['hr_ratio'] = df['thalach']/df['age']
    if 'age' in df.columns:
        df['age_bin'] = pd.cut(
            df['age'], bins=5, labels=False
        ).astype('category')
        
    return df

# BaseEstimator --> Provides utility methods like get_params() and set_params().
# TransformerMixin --> Implements a default fit_transform() method based on fit() and transform()
class AddNewFeaturesTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        self.columns_ = X.columns
        self.new_features_ = []
        if {'chol', 'age'} <= set(df.columns):
            self.new_features_.append('chol_per_age')
        if {'trestbps', 'age'} <= set(df.columns):
            self.new_features_.append('bps_per_age')
        if {'thalach', 'age'} <= set(df.columns):
            self.new_features_.append('hr_ratio')
        if 'age' in df.columns:
            self.new_features_.append('age_bin')
            
        return self
    
    def transform(self, X):
        return add_new_features_func(X)
    
    def get_feature_names_out(self, input_features=None):
        return list(self.columns_) + self.new_features_

gen_num = ['chol_per_age', 'bps_per_age', 'hr_ratio']
gen_cat = ['age_bin']
all_nums = [c for c in numeric_cols]  + gen_num
all_cats = [c for c in categorical_cols] + gen_cat

num_proc = Pipeline([('imp', SimpleImputer(strategy='median')),
                        ('sc', StandardScaler())])
cat_proc = Pipeline([('imp', SimpleImputer(strategy='most_frequent')),
                        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])

pre = ColumnTransformer([
    ('num', num_proc, all_nums),
    ('cat', cat_proc, all_cats)
], verbose_feature_names_out=False).set_output(transform='pandas')

fe_pre = Pipeline([
    ('add', AddNewFeaturesTransformer()),
    ('pre', pre),
]).set_output(transform='pandas')

Xt_tr = fe_pre.fit_transform(X_train, y_train)
Xt_va = fe_pre.transform(X_val)
Xt_te = fe_pre.transform(X_test)

nz_cols = Xt_tr.columns[Xt_tr.nunique(dropna=False) > 1]
Xt_tr = Xt_tr[nz_cols]
Xt_va = Xt_va.reindex(columns=Xt_tr.columns, fill_value=0)
Xt_te = Xt_te.reindex(columns=Xt_tr.columns, fill_value=0)

# Apply mutual infor to evaluate and select the features that have the strongest relationship with the target variable

ohe = fe_pre.named_steps['pre'].named_transformers_['cat'].named_steps['ohe']
cat_names = list(ohe.get_feature_names_out(all_cats))
is_discrete = np.array(
    [c in cat_names for c in Xt_tr.columns],
    dtype=bool,
)
mi = mutual_info_classif(Xt_tr.values, y_train.values,
                         discrete_features=is_discrete,
                         random_state=42)

mi_series = pd.Series(
    mi, index=Xt_tr.columns
).sort_values(ascending=False)

N = min(20, len(mi_series))
topN = mi_series.head(N).iloc[::-1]
plt.figure(figsize=(10, max(6, 0.35*N)))
plt.barh(topN.index, topN.values)
plt.title('Top MI scores (Train)')
plt.xlabel('MI score')
plt.ylabel('Feature')
plt.tight_layout()
plt.savefig('top_mi_scores.png', bbox_inches='tight')
plt.show()

K = df.columns.drop('target').shape[0]
topk_cols = list(mi_series.head(K).index)
fe_tr = Xt_tr[topk_cols].assign(target=y_train.values)
fe_va = Xt_va[topk_cols].assign(target=y_val.values)
fe_te = Xt_te[topk_cols].assign(target=y_test.values)

out = Path('splits'); out.mkdir(parents=True, exist_ok=True)
fe_tr.to_csv(out/'fe_train.csv', index=False)
fe_va.to_csv(out/'fe_val.csv', index=False)
fe_te.to_csv(out/'fe_test.csv', index=False)

# Using Decision Tree again to select the most common  deatures from dataset via FE method.
dt_fe_feature_selection_pipeline = Pipeline([
    ('preprocess', fe_pre),
    ('decision_tree', DecisionTreeClassifier(random_state=42))
])

dt_fe_feature_selection_pipeline.fit(X_train, y_train)
pipeline_feature_names = dt_fe_feature_selection_pipeline.named_steps['preprocess'].get_feature_names_out()

pipeline_importance_series = pd.Series(
    dt_fe_feature_selection_pipeline.named_steps['decision_tree'].feature_importances_,
    index=pipeline_feature_names
)

sorted_feature_importances = pipeline_importance_series.sort_values(ascending=False)

selected_features = sorted_feature_importances.head(K_features).index.tolist()
print(f"Top {K_features} selected features: {selected_features}")

X_fe_dt_train = Xt_tr[selected_features]
X_fe_dt_val = Xt_va[selected_features]
X_fe_dt_test = Xt_te[selected_features]

pd.concat([X_fe_dt_train, y_train.rename(TARGET)],
  axis=1).to_csv(out_dir / 'fe_dt_train.csv', index=False)
pd.concat([X_fe_dt_val, y_val.rename(TARGET)],
  axis=1).to_csv(out_dir / 'fe_dt_val.csv', index=False)
pd.concat([X_fe_dt_test, y_test.rename(TARGET)],
  axis=1).to_csv(out_dir / 'fe_dt_test.csv', index=False)

plt.style.use('seaborn-v0_8-whitegrid')
plt.figure(figsize=(10, 6))

# Create the bar plot using seaborn
# The 'palette="copper"' argument creates the brown-to-orange color gradient
sns.barplot(x=sorted_feature_importances.index, y=sorted_feature_importances.values, palette="Oranges_r")

# Add a title and labels for the axes
plt.title('Feature Importance Scores', fontsize=14)
plt.xlabel('Features', fontsize=12)
plt.ylabel('Feature Importance Score', fontsize=12)

# Display the plot
plt.savefig(f"chart_importance_feature_2.png", dpi=150, bbox_inches="tight")

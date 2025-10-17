import os 
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings('ignore')

SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
np.random.seed(SEED)
random.seed(SEED)
print(f'Seed: {SEED}')

# Read data from file.csv
def read_csv(file_path):
    df = pd.read_csv(file_path)
    X = df.drop('target', axis=1)
    y = df['target']
    
    # print('Shape df: ', df.shape)
    # print('Shape X: ', X.shape)
    # print('Shape y: ', y.shape)
    
    return X, y

# Read data
# Original Dataset
X_train, y_train = read_csv('splits/raw_train.csv')
X_val, y_val = read_csv('splits/raw_val.csv')
X_test, y_test = read_csv('splits/raw_test.csv')

#FE dataset
X_fe_train, y_fe_train = read_csv('splits/fe_train.csv')
X_fe_val, y_fe_val = read_csv('splits/fe_val.csv')
X_fe_test, y_fe_test = read_csv('splits/fe_test.csv')

# Original + DT Dataset
X_dt_train, y_dt_train = read_csv('splits/dt_train.csv')
X_dt_val, y_dt_val = read_csv('splits/dt_val.csv')
X_dt_test, y_dt_test = read_csv('splits/dt_test.csv')

 # FE + DT Dataset
X_fe_dt_train, y_fe_dt_train = read_csv('splits/fe_dt_train.csv')
X_fe_dt_val, y_fe_dt_val = read_csv('splits/fe_dt_val.csv')
X_fe_dt_test, y_fe_dt_test = read_csv('splits/fe_dt_test.csv')


# Find the optimal quantity of subtrees for RF model by Stratifiel K-Fold Cross-Validation
def find_optimal_rf(
    X_train, y_train, n_estimators_range=range(50,501,50), cv_splits=3,
    max_depth=5,min_samples_split=2,min_samples_leaf=1,
    max_features='sqrt',bootstrap=True,class_weight=None
):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
    scores = []
    for n in n_estimators_range:
        rf = RandomForestClassifier(
            n_estimators=n, max_depth=max_depth, min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,max_features=max_features,
            bootstrap=bootstrap,class_weight=class_weight,n_jobs=-1, random_state=SEED
        )
        cv_score = cross_val_score(rf, X_train, y_train, 
                                cv=cv,scoring="accuracy",n_jobs=-1)
        scores.append(cv_score.mean())
    plt.figure(figsize=(10, 6))
    plt.plot(list(n_estimators_range), scores, 'bo-')
    plt.title(f'Chọn n_estimators tối ưu cho Random Forest (CV={cv_splits}-fold)')
    plt.xlabel('n_estimators')
    plt.ylabel('Cross-Validation Accuracy')
    plt.grid(True)
    plt.savefig('Random_Forest_accuracy_and_n_estimators.png', bbox_inches='tight')
    plt.show()
    
    best_n = list(n_estimators_range)[int(np.argmax(scores))]
    print(f'n_estimators tối ưu (CV): {best_n}')
    
    best_model = RandomForestClassifier(
        n_estimators=best_n, max_depth=max_depth,min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,max_features=max_features,bootstrap=bootstrap,
        class_weight=class_weight,n_jobs=-1,random_state=SEED
    )
    best_model.fit(X_train, y_train)
    return best_model, best_n, max(scores)

# Next, another function trains, validates, and tests the model’s performance
def evaluate_val_rf(X_train, y_train, X_val, y_val,
                    n_estimators_range=range(50,501,50), cv_splits=5,max_depth=5,
                    min_samples_split=2,min_samples_leaf=1,max_features='sqrt',
                    bootstrap=True,class_weight=None
                    ):
    print('Find n_estimators optimize for the RF')
    rf_model, best_n, cv_acc = find_optimal_rf(
        X_train, y_train, n_estimators_range=n_estimators_range, cv_splits=cv_splits,
        max_depth=max_depth,min_samples_split=min_samples_split,min_samples_leaf=min_samples_leaf,
        max_features=max_features, bootstrap=bootstrap, class_weight=class_weight
    )
    
    val_pred = rf_model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    print(f'\nAccuracy Random Forest on validation: {val_acc:.4f}')
    print('Classification Report:')
    print(classification_report(y_val, val_pred))
    return rf_model, val_acc, {'n_estimators': best_n}

def evaluate_test_rf(rf_model, X_test, y_test):
    test_pred = rf_model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    print(f'\nAccuracy Random Forest on validation: {test_acc:.4f}')
    print('Classification Report:')
    print(classification_report(y_test, test_pred))
    return test_acc
    
    
# Carrying training model on four dif datasets

#RF on Original Dataset
rf_model, val_acc, best_params = evaluate_val_rf(
    X_train, y_train, X_val, y_val
)
test_acc = evaluate_test_rf(rf_model, X_test, y_test)

#RF on Feature Engineering Dataset
rf_model, val_fe_acc, best_params =evaluate_val_rf(
    X_fe_train, y_fe_train, X_fe_val, y_fe_val
)
test_fe_acc = evaluate_test_rf(rf_model, X_fe_test, y_fe_test)

# RF on Original DT Dataset
rf_model, val_dt_acc, best_params = evaluate_val_rf(
    X_dt_train, y_dt_train, X_dt_val, y_dt_val
)
test_dt_acc = evaluate_test_rf(rf_model, X_dt_test, y_dt_test)

# RF on Feature Engineering DT Dataset
rf_model, val_fe_dt_acc, best_params = evaluate_val_rf(
    X_fe_dt_train, y_fe_dt_train, X_fe_dt_val, y_fe_dt_val
)
test_fe_dt_acc = evaluate_test_rf(rf_model, X_fe_dt_test, y_fe_dt_test)

# Plot a chart comparing the model’s performance across datasets.

plt.rcParams['font.family'] = 'Serif'

labels = ['Original', 'FE', 'Original + DT', 'FE + DT']
val_accs = [val_acc, val_fe_acc, val_dt_acc, val_fe_dt_acc]
test_accs = [test_acc, test_fe_acc, test_dt_acc, test_fe_dt_acc]

x = np.arange(len(labels))
width = 0.3

fig, ax = plt.subplots(figsize=(5,5))

rects1 = ax.bar(x-width/2, val_accs, width,
                label='Validation Accuracy',
                color='tab:blue', edgecolor='black', linewidth=1.2)
rects2 = ax.bar(x+width/2, test_accs, width,
                label='Test Accuracy',
                color='tab:red', edgecolor='black', linewidth=1.2)

ax.set_ylim(0.5, 1.05)
ax.set_ylabel('Accuracy')
ax.set_title('Random Forest', fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend(ncol=2, loc='upper center')


def autolabel(rects):
    for rect in rects:
       h = rect.get_height()
       ax.annotate(f'{h:.2f}', xy=(rect.get_x()+rect.get_width()/2, h),
       ha='center', va='bottom')
autolabel(rects1)
autolabel(rects2)
fig.tight_layout()
plt.savefig('rf_performance_4datasets.png', bbox_inches='tight')
plt.show()
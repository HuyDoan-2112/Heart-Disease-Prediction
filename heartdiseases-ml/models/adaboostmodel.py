import os 
import random
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
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


# Find the optimal subtrees by Stratified K-Fold Cross-Validation

def find_optimal_ada(
    X_train, y_train,
    n_estimators_range=range(50, 501, 50),
    cv_splits=3,
    learning_rate=0.1,
    base_max_depth=1,
    algorithm='SAMME'
):
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=SEED)
    scores = []
    
    for n in n_estimators_range:
        ada  = AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=base_max_depth, random_state=SEED),
            n_estimators=n, learning_rate=learning_rate,
            algorithm=algorithm, random_state=SEED
        )
        
        cv_score = cross_val_score(
            ada, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=-1
        )
        scores.append(cv_score.mean())
    
    plt.figure(figsize=(10,6))
    plt.plot(list(n_estimators_range), scores, 'bo-')
    plt.title(f'Choose n_estimators optimize for AdaBoost(CV={cv_splits}-fold)')
    plt.xlabel('n_estimators')
    plt.ylabel('Cross-Validation Accuracy')
    plt.grid(True)
    plt.show()
    
    best_n = list(n_estimators_range)[int(np.argmax(scores))]
    print(f'n_estimators optimizes (CV): {best_n}')
    
    best_model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=base_max_depth, random_state=SEED),
        n_estimators=best_n, learning_rate=learning_rate,
        algorithm=algorithm, random_state=SEED
    )
    best_model.fit(X_train, y_train)
    return best_model, best_n, max(scores)

# Create the functions to evaluate and train the AdaBoost model
def evaluate_val_ada(X_train, y_train, X_val, y_val,
                     n_estimators_range=range(50, 501, 50),
                     cv_splits=3,
                     learning_rate=0.1,
                     base_max_depth=1,
                     algorithm='SAMME'):
    print('Find n_estimators optimize for Adaboost')
    ada_model, best_n, cv_acc = find_optimal_ada(
        X_train, y_train, 
        n_estimators_range=n_estimators_range,
        cv_splits=cv_splits,
        learning_rate=learning_rate,
        base_max_depth=base_max_depth,
        algorithm=algorithm
    )
    
    val_pred = ada_model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)
    print(f'\n The accuracy AdaBoost: {val_acc:.4f}')
    print('Classification Report: ')
    print(classification_report(y_val, val_pred))
    return ada_model, val_acc, {"n_estimators": best_n}

def evaluate_test_ada(ada_model, X_test, y_test):
    test_pred = ada_model.predict(X_test)
    test_acc = accuracy_score(y_test, test_pred)
    print(f'\nThe accuracy Adaboost on test dataset: {test_acc:.4f}')
    print('Classification Report: ')
    print(classification_report(y_test, test_pred))
    return test_acc


  
# Carrying training model on four dif datasets

#ADA on Original Dataset
rf_model, val_acc, best_params = evaluate_val_ada(
    X_train, y_train, X_val, y_val
)
test_acc = evaluate_test_ada(rf_model, X_test, y_test)

#ADA on Feature Engineering Dataset
rf_model, val_fe_acc, best_params =evaluate_val_ada(
    X_fe_train, y_fe_train, X_fe_val, y_fe_val
)
test_fe_acc = evaluate_test_ada(rf_model, X_fe_test, y_fe_test)

# ADA on Original DT Dataset
rf_model, val_dt_acc, best_params = evaluate_val_ada(
    X_dt_train, y_dt_train, X_dt_val, y_dt_val
)
test_dt_acc = evaluate_test_ada(rf_model, X_dt_test, y_dt_test)

# ADA on Feature Engineering DT Dataset
rf_model, val_fe_dt_acc, best_params = evaluate_val_ada(
    X_fe_dt_train, y_fe_dt_train, X_fe_dt_val, y_fe_dt_val
)
test_fe_dt_acc = evaluate_test_ada(rf_model, X_fe_dt_test, y_fe_dt_test)

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
ax.set_title('AdaBoost', fontsize=16)
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
plt.savefig('Plots/ada_performance_4datasets.png', bbox_inches='tight')
plt.show()
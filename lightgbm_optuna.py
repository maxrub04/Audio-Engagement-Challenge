# -*- coding: utf-8 -*-
import re
import gc
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import optuna
from optuna.integration import LightGBMPruningCallback
import matplotlib.pyplot as plt

RANDOM_STATE = 42

# -------------------------
# 0. Load Data
# -------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
print("train:", train.shape, "test:", test.shape)

# -------------------------
# 1. Basic processing and imputation
# -------------------------
# Handle missing values with median imputation and create flags
missing_values_columns = ["Episode_Length_minutes", "Number_of_Ads"]
for col in missing_values_columns:
    train[f'{col}_IS_IMPUTED'] = train[col].isnull().astype(int)
    test[f'{col}_IS_IMPUTED'] = test[col].isnull().astype(int)
    med = train[col].median()
    train[col] = train[col].fillna(med)
    test[col] = test[col].fillna(med)

# Fill missing popularity values with 0
train['Guest_Popularity_percentage'] = train['Guest_Popularity_percentage'].fillna(0.0)
test['Guest_Popularity_percentage'] = test['Guest_Popularity_percentage'].fillna(0.0)
train['Host_Popularity_percentage'] = train['Host_Popularity_percentage'].fillna(0.0)
test['Host_Popularity_percentage'] = test['Host_Popularity_percentage'].fillna(0.0)

# Fill missing Episode_Title for TF-IDF
train['Episode_Title'] = train['Episode_Title'].fillna('missing').astype(str)
test['Episode_Title'] = test['Episode_Title'].fillna('missing').astype(str)

# -------------------------
# 2. Feature Engineering
# -------------------------
eps = 1e-6
for df in (train, test):
    df['Popularity_Combined'] = df.get('Host_Popularity_percentage', 0.0) + df.get('Guest_Popularity_percentage', 0.0)
    df['Ads_per_Minute'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + eps)
    df['Ads_per_Minute'] = df['Ads_per_Minute'].fillna(0.0)
    df['Len_div_ads'] = df['Episode_Length_minutes'] / (df['Number_of_Ads'] + 1.0)
    df['Has_Guest'] = (df.get('Guest_Popularity_percentage', 0.0) > 0).astype(int)

# Extract episode number from title
def extract_episode_number_safe(title):
    if pd.isna(title):
        return 0
    m = re.search(r'(\d{1,5})\b(?!.*\d)', str(title))
    return int(m.group(1)) if m else 0

train['Episode_Number'] = train['Episode_Title'].apply(extract_episode_number_safe)
test['Episode_Number'] = test['Episode_Title'].apply(extract_episode_number_safe)

# Create ad groups
bins = [-1, 1, 2, 104]  # include -1 so 0 goes to first bin
labels = ['0-1', '1-2', '2+']
train['Ads_Groups'] = pd.cut(train['Number_of_Ads'], bins=bins, labels=labels)
test['Ads_Groups'] = pd.cut(test['Number_of_Ads'], bins=bins, labels=labels)
train['Ads_Groups'] = train['Ads_Groups'].fillna('0-1')
test['Ads_Groups'] = test['Ads_Groups'].fillna('0-1')

# -------------------------
# 3. Categorical Features Encoding
# -------------------------
from sklearn.preprocessing import LabelEncoder
categorical_small = ['Genre', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment', 'Ads_Groups']
for col in categorical_small:
    if col in train.columns:
        le = LabelEncoder()
        all_values = pd.concat([train[col].astype(str), test[col].astype(str)], axis=0)
        le.fit(all_values)
        train[col + '_le'] = le.transform(train[col].astype(str))
        test[col + '_le'] = le.transform(test[col].astype(str))

# -------------------------
# 4. TF-IDF Features
# -------------------------
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2), min_df=3)
all_titles = pd.concat([train['Episode_Title'], test['Episode_Title']], axis=0)
tfidf.fit(all_titles)
X_tfidf_train = tfidf.transform(train['Episode_Title'])
X_tfidf_test = tfidf.transform(test['Episode_Title'])

# -------------------------
# 5. Feature Selection
# -------------------------
tab_features = [
    'Episode_Length_minutes', 'Episode_Length_minutes_IS_IMPUTED',
    'Guest_Popularity_percentage', 'Host_Popularity_percentage',
    'Popularity_Combined', 'Ads_per_Minute', 'Len_div_ads',
    'Episode_Number', 'Has_Guest'
]
tab_features += [c + '_le' for c in categorical_small]
tab_features = [c for c in tab_features if c in train.columns]

X_tab_train = train[tab_features].reset_index(drop=True)
X_tab_test = test[tab_features].reset_index(drop=True)
y = train['Listening_Time_minutes'].values

# -------------------------
# 6. Combine Features
# -------------------------
X_tab_train_sparse = sparse.csr_matrix(X_tab_train.values)
X_tab_test_sparse = sparse.csr_matrix(X_tab_test.values)
X_train_full = sparse.hstack([X_tab_train_sparse, X_tfidf_train]).tocsr()
X_test_full = sparse.hstack([X_tab_test_sparse, X_tfidf_test]).tocsr()

del X_tab_train_sparse, X_tab_test_sparse, X_tfidf_train, X_tfidf_test
gc.collect()

# -------------------------
# 7. Train/Validation Split
# -------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y,
    test_size=0.2,
    random_state=RANDOM_STATE
)

print("X_train shape:", X_train.shape, "X_val shape:", X_val.shape)
print("y_train shape:", y_train.shape, "y_val shape:", y_val.shape)

# -------------------------
# 8. Optuna Optimization
# -------------------------
def objective(trial):
    """Optuna objective function for LightGBM optimization"""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'n_estimators': 10000,
        'random_state': RANDOM_STATE,

        # Parameters to optimize
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
    }

    model = lgb.LGBMRegressor(**params)

    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            LightGBMPruningCallback(trial, 'rmse')
        ]
    )

    # Evaluate
    val_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))

    return rmse

# Create and run Optuna study
study = optuna.create_study(
    direction='minimize',
    study_name='lightgbm_optimization',
    pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
)

n_trials = 100
study.optimize(objective, n_trials=n_trials)

# Print optimization results
print("\nBest parameters:", study.best_params)
print(f"Best RMSE: {study.best_value:.6f}")

# -------------------------
# 9. Train Final Model
# -------------------------
best_params = study.best_params
best_params.update({
    'n_estimators': 40000,
    'random_state': RANDOM_STATE,
    'objective': 'regression',
    'metric': 'rmse'
})

final_model = lgb.LGBMRegressor(**best_params)
final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    eval_metric='rmse',
    callbacks=[lgb.early_stopping(stopping_rounds=100)]
)

# -------------------------
# 10. Make Predictions
# -------------------------
test_pred = final_model.predict(X_test_full, num_iteration=final_model.best_iteration_)

if 'id' in test.columns:
    sub = pd.DataFrame({
        'id': test['id'].values,
        'Listening_Time_minutes': test_pred
    })
    sub.to_csv('submission_optuna.csv', index=False)
    print("Saved submission_optuna.csv")

# -------------------------
# 11. Visualization
# -------------------------
plt.figure(figsize=(10, 6))
optuna.visualization.matplotlib.plot_optimization_history(study)
plt.title('Optimization History')
plt.show()

plt.figure(figsize=(10, 6))
optuna.visualization.matplotlib.plot_param_importances(study)
plt.title('Parameter Importances')
plt.show()

# Clean up
del X_train_full, X_test_full, X_train, X_val
gc.collect()
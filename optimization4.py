import re
import gc
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

RANDOM_STATE = 0

# -------------------------
# 0. Load
# -------------------------
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")
print("train:", train.shape, "test:", test.shape)

# -------------------------
# 1. Basic processing and imputation
# -------------------------
# Pass flags + median (calculated on train)
missing_values_columns = ["Episode_Length_minutes", "Number_of_Ads"]
for col in missing_values_columns:
    train[f'{col}_IS_IMPUTED'] = train[col].isnull().astype(int)
    test[f'{col}_IS_IMPUTED'] = test[col].isnull().astype(int)
    med = train[col].median()
    train[col] = train[col].fillna(med)
    test[col]  = test[col].fillna(med)

# Guest/host popularity
train['Guest_Popularity_percentage'] = train['Guest_Popularity_percentage'].fillna(0.0)
test['Guest_Popularity_percentage'] = test['Guest_Popularity_percentage'].fillna(0.0)
train['Host_Popularity_percentage'] = train['Host_Popularity_percentage'].fillna(0.0)
test['Host_Popularity_percentage']  = test['Host_Popularity_percentage'].fillna(0.0)

# Episode_Title: fill in the blank rows for TF-IDF
train['Episode_Title'] = train['Episode_Title'].fillna('missing').astype(str)
test['Episode_Title']  = test['Episode_Title'].fillna('missing').astype(str)

# -------------------------
# 2. Derived features
# -------------------------
eps = 1e-6
for df in (train, test):
    df['Popularity_Combined'] = (df.get('Host_Popularity_percentage', 0.0) + df.get('Guest_Popularity_percentage', 0.0))
    df['Ads_per_Minute'] = df['Number_of_Ads'] / (df['Episode_Length_minutes'] + eps)
    df['Ads_per_Minute'] = df['Ads_per_Minute'].fillna(0.0)
    df['Len_div_ads'] = df['Episode_Length_minutes'] / (df['Number_of_Ads'] + 1.0)
    df['Has_Guest'] = (df.get('Guest_Popularity_percentage', 0.0) > 0).astype(int)

def extract_episode_number_safe(title):
    if pd.isna(title):
        return 0
    m = re.search(r'(\d{1,5})\b(?!.*\d)', str(title))
    return int(m.group(1)) if m else 0

train['Episode_Number'] = train['Episode_Title'].apply(extract_episode_number_safe)
test['Episode_Number']  = test['Episode_Title'].apply(extract_episode_number_safe)

# Ads_Groups: binnin
bins = [ -1, 1, 2, 104 ]  # include -1 so 0 goes to first bin
labels = ['0-1', '1-2', '2+']
train['Ads_Groups'] = pd.cut(train['Number_of_Ads'], bins=bins, labels=labels)
test['Ads_Groups']  = pd.cut(test['Number_of_Ads'], bins=bins, labels=labels)
train['Ads_Groups'] = train['Ads_Groups'].fillna('0-1')
test['Ads_Groups']  = test['Ads_Groups'].fillna('0-1')

# -------------------------
# 3. Category processing (LabelEncoder for small columns) — LightGBM can handle categorical data, but when using sparse TF-IDF, we will pass a clean matrix,
#    so we encode as int for tabular categories and use TF-IDF separately.
# -------------------------
from sklearn.preprocessing import LabelEncoder
categorical_small = ['Genre','Publication_Day','Publication_Time','Episode_Sentiment','Ads_Groups']
for col in categorical_small:
    if col in train.columns:
        le = LabelEncoder()
        all_values = pd.concat([train[col].astype(str), test[col].astype(str)], axis=0)
        le.fit(all_values)
        train[col + '_le'] = le.transform(train[col].astype(str))
        test[col + '_le']  = le.transform(test[col].astype(str))

# -------------------------
# 4. TF-IDF for Episode_Title
# -------------------------
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1,2), min_df=3)
all_titles = pd.concat([train['Episode_Title'], test['Episode_Title']], axis=0)
tfidf.fit(all_titles)
X_tfidf_train = tfidf.transform(train['Episode_Title'])
X_tfidf_test  = tfidf.transform(test['Episode_Title'])

# -------------------------
# 5.final features
# -------------------------
tab_features = [
    'Episode_Length_minutes','Episode_Length_minutes_IS_IMPUTED',
    'Guest_Popularity_percentage','Host_Popularity_percentage',
    'Popularity_Combined','Ads_per_Minute','Len_div_ads',
    'Episode_Number','Has_Guest'
]
# add coded categorical
tab_features += [c + '_le' for c in categorical_small]

# Ensure that all columns are present
tab_features = [c for c in tab_features if c in train.columns]

X_tab_train = train[tab_features].reset_index(drop=True)
X_tab_test  = test[tab_features].reset_index(drop=True)
y = train['Listening_Time_minutes'].values  # target at original scale

# -------------------------
# 6. Combining sparse TF-IDF and tabular features
# -------------------------
X_tab_train_sparse = sparse.csr_matrix(X_tab_train.values)
X_tab_test_sparse  = sparse.csr_matrix(X_tab_test.values)

X_train_full = sparse.hstack([X_tab_train_sparse, X_tfidf_train]).tocsr()
X_test_full  = sparse.hstack([X_tab_test_sparse, X_tfidf_test]).tocsr()

del X_tab_train_sparse, X_tab_test_sparse, X_tfidf_train, X_tfidf_test
gc.collect()

import optuna
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from optuna.integration import LightGBMPruningCallback
import numpy as np


def objective(trial):
    params = {
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'num_leaves': trial.suggest_int('num_leaves', 2, 256),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
    }

    model = lgb.LGBMRegressor(
        random_state=0,
        n_estimators=10_000,
        force_col_wise=True,
        **params
    )

    model.fit(
        x, y,
        eval_set=[(x_val, y_val)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(stopping_rounds=300, verbose=False),
            LightGBMPruningCallback(trial, "rmse")
        ]
    )

    preds = model.predict(x_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return rmse



x, x_val, y, y_val = train_test_split(X_train_full, y, random_state=0,
test_size=0.2)
x, x_test, y, y_test = train_test_split(x, y, random_state=0, test_size=0.25)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

print(study.best_value)
print(study.best_params)
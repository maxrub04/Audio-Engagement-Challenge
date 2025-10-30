import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer, mean_squared_error
from catboost import CatBoostRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from skopt.callbacks import DeltaYStopper, DeadlineStopper
from functools import partial
from time import time
# Stratified CV
from sklearn.model_selection import KFold
import pprint
import warnings
warnings.filterwarnings("ignore")



def neg_root_mean_squared_error(y_true, y_pred, **kwargs):
    """Return the negative RMSE (for minimization). Ignores unexpected kwargs."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    return -rmse  # negative so that higher = better for BayesSearchCV


def report_perf(optimizer, X, y, title="CatBoost", callbacks=None):
    start = time()
    if callbacks is not None:
        optimizer.fit(X, y, callback=callbacks)
    else:
        optimizer.fit(X, y)
    d = pd.DataFrame(optimizer.cv_results_)
    best_score = optimizer.best_score_
    best_score_std = d.iloc[optimizer.best_index_].std_test_score
    best_params = optimizer.best_params_

    print(
        f"{title} optimization took {time() - start:.2f} seconds, best CV score: {best_score:.4f} ± {best_score_std:.4f}")
    print("Best parameters:")
    pprint.pprint(best_params)

    return best_params

df_train = pd.read_csv("../train.csv")
df_test = pd.read_csv("../test.csv")

#Delete duplicates
df_train = df_train.drop_duplicates()#1

y = df_train["Listening_Time_minutes"]
X = df_train.drop(columns=["Listening_Time_minutes", "id"])
X_test = df_test.drop(columns=["id"])

#Handaling outliers
if "Guest_Popularity_percentage" in X.columns:
    X["Guest_Popularity_percentage"] = X["Guest_Popularity_percentage"].clip(upper=100)

if "Host_Popularity_percentage" in X.columns:
    X["Host_Popularity_percentage"] = X["Host_Popularity_percentage"].clip(lower=20, upper=100)

if "Episode_Length_minutes" in X.columns:
    X["Episode_Length_minutes"] = X["Episode_Length_minutes"].clip(lower=5, upper=120)

if "Number_of_Ads" in X.columns:#2
    X["Number_of_Ads"] = X["Number_of_Ads"].clip(upper=3)

cat_features = ["Podcast_Name", "Episode_Title", "Genre", "Publication_Day", "Publication_Time", "Episode_Sentiment"]

# Stratified proxy for regression
bins = int(np.floor(1 + np.log2(len(X))))
y_binned = pd.cut(y, bins=bins, labels=False)

kf = KFold(n_splits=20, shuffle=True, random_state=42)

# RMSE scorer (we want to minimize RMSE)
#scoring = make_scorer(partial(mean_squared_error, squared=False), greater_is_better=False)
scoring = make_scorer(neg_root_mean_squared_error, greater_is_better=True)

# Base CatBoost regressor (no hyperparameters fixed)
cat_model = CatBoostRegressor(
    loss_function='RMSE',
    cat_features=cat_features,
    verbose=False,
    random_state=42
)

search_spaces = {
    'iterations': Integer(10, 1000),
    'depth': Integer(1, 8),
    'learning_rate': Real(0.01, 1.0, 'log-uniform'),
    'l2_leaf_reg': Integer(2, 30),
    'border_count': Integer(1, 255),
    'random_strength': Real(1e-9, 10.0, 'log-uniform'),
    'bagging_temperature': Real(0.0, 1.0, 'uniform'),
}

opt = BayesSearchCV(
    estimator=cat_model,
    search_spaces=search_spaces,
    scoring=scoring,
    cv=kf,
    n_iter=50,            # Number of hyperparameter trials
    n_jobs=-1,
    refit=False,
    optimizer_kwargs={'base_estimator': 'GP'},
    random_state=42
)

# Early stopping and time limit
delta_stop = DeltaYStopper(delta=0.0001)
time_stop = DeadlineStopper(total_time=60 * 60 * 8)  # 8 hours

best_params = report_perf(opt, X, y, callbacks=[delta_stop, time_stop])


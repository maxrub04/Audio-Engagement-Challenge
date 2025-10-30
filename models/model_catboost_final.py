from catboost import CatBoostRegressor
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from TargetEncode import TargetEncode

# Load data
df_train = pd.read_csv("../train.csv")
df_test = pd.read_csv("../test.csv")

#Delete duplicates
#df_train = df_train.drop_duplicates()#1

# Target and features
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


# 🔹 Regular K-Fold
kf = KFold(n_splits=20, shuffle=True, random_state=0)

rmse_scores = []
test_preds = np.zeros(len(X_test))  # store averaged predictions

for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")

    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    model = CatBoostRegressor(
        iterations=323,
        learning_rate= 0.0673344419215237,
        depth=8,
        loss_function='RMSE',
        cat_features=cat_features,
        verbose=False,
        random_state=0,
        bagging_temperature=  0.41010395885331385,
        border_count= 186,
        l2_leaf_reg= 21,
        random_strength=  3.230824361824754e-06
     )

    model.fit(X_train, y_train, eval_set=(X_valid, y_valid),
              early_stopping_rounds=50, verbose=False)

    # Validation RMSE
    preds_valid = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, preds_valid))
    rmse_scores.append(rmse)
    print(f"Fold {fold + 1} RMSE: {rmse:.4f}")

    # Add fold test predictions (averaged)
    test_preds += model.predict(X_test) / kf.n_splits

print("\nMean CV RMSE:", np.mean(rmse_scores))
"""
======================================================================
 Bank Customer Churn Prediction - Full Pipeline
======================================================================
Dataset : Churn_Modelling.csv (10,000 customers, 14 columns)
Target  : Exited (1 = customer left the bank, 0 = customer stayed)

Steps:
    1) Load data + quick EDA
    2) Clean data + feature engineering
    3) Train/Test split
    4) Hyperparameter tuning for multiple models
    5) Evaluate the best model + save it

IMPORTANT NOTE ABOUT ACCURACY:
    This dataset has class imbalance (~80% stayed, ~20% churned).
    A "dumb" model that always predicts "stayed" would already get
    79.6% accuracy without learning anything. The best results
    reported on this dataset (Kaggle / public benchmarks) typically
    fall between 85% and 87% with strong models (XGBoost/RandomForest)
    even after full hyperparameter tuning, because the available
    features simply don't fully explain every churn decision.
    If a model reports 90%+ accuracy on this exact dataset, it's very
    likely due to data leakage or overfitting (e.g. using
    CustomerId/Surname, or leaking future information).
    This script aims for the best REALISTIC accuracy (~87%) and
    reports a full set of metrics (Precision/Recall/F1/ROC-AUC) so
    the picture is clearer than just one number.
======================================================================
"""

import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from xgboost import XGBClassifier
import joblib

RANDOM_STATE = 42
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else "Churn_Modelling.csv"


# ----------------------------------------------------------------
# 1) Load data
# ----------------------------------------------------------------
def load_data(path):
    df = pd.read_csv(path)
    print(f"Data shape: {df.shape}")
    print("\nTarget distribution (Exited):")
    print(df["Exited"].value_counts(normalize=True).round(4) * 100)
    print(f"\nNumber of missing values: {df.isnull().sum().sum()}")
    return df


# ----------------------------------------------------------------
# 2) Cleaning + feature engineering
# ----------------------------------------------------------------
def feature_engineering(df):
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])  # ID columns have no predictive value

    # New derived features (usually give a small accuracy boost)
    df["BalanceSalaryRatio"] = df["Balance"] / (df["EstimatedSalary"] + 1)
    df["TenureByAge"] = df["Tenure"] / df["Age"]
    df["CreditScoreGivenAge"] = df["CreditScore"] / df["Age"]
    df["IsZeroBalance"] = (df["Balance"] == 0).astype(int)
    df["ProductsPerTenure"] = df["NumOfProducts"] / (df["Tenure"] + 1)

    # Encode categorical columns
    df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)

    X = df.drop(columns=["Exited"])
    y = df["Exited"]
    return X, y


# ----------------------------------------------------------------
# 3) Hyperparameter tuning
# ----------------------------------------------------------------
def tune_xgboost(X_train, y_train):
    param_dist = {
        "n_estimators": [200, 300, 400, 500, 600],
        "max_depth": [3, 4, 5, 6, 7],
        "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
        "subsample": [0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5, 7],
        "gamma": [0, 0.1, 0.2, 0.3],
    }
    xgb = XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        xgb, param_dist, n_iter=50, scoring="accuracy",
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")
    return search.best_estimator_


def tune_random_forest(X_train, y_train):
    param_dist = {
        "n_estimators": [200, 300, 400, 500],
        "max_depth": [6, 8, 10, 12, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2"],
        "class_weight": [None, "balanced"],
    }
    rf = RandomForestClassifier(random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        rf, param_dist, n_iter=40, scoring="accuracy",
        cv=StratifiedKFold(5, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)
    print(f"Best CV accuracy: {search.best_score_:.4f}")
    print(f"Best params: {search.best_params_}")
    return search.best_estimator_


# ----------------------------------------------------------------
# 4) Model evaluation
# ----------------------------------------------------------------
def evaluate(model, X_test, y_test, name):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred)
    rec = recall_score(y_test, pred)
    f1 = f1_score(y_test, pred)
    auc = roc_auc_score(y_test, proba)

    print(f"\n{'=' * 55}")
    print(f"Results for: {name}")
    print(f"{'=' * 55}")
    print(f"Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"Precision : {prec:.4f}")
    print(f"Recall    : {rec:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print(f"ROC-AUC   : {auc:.4f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, pred))
    print("\nClassification Report:")
    print(classification_report(y_test, pred))

    return acc


# ----------------------------------------------------------------
# Full run
# ----------------------------------------------------------------
def main():
    df = load_data(DATA_PATH)
    X, y = feature_engineering(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Scale numeric columns (useful if you add other models, not strictly
    # needed for tree-based models like XGBoost/RandomForest)
    scaler = StandardScaler()
    num_cols = ["CreditScore", "Age", "Tenure", "Balance", "NumOfProducts",
                "EstimatedSalary", "BalanceSalaryRatio", "TenureByAge",
                "CreditScoreGivenAge", "ProductsPerTenure"]
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_test_scaled[num_cols] = scaler.transform(X_test[num_cols])

    print("\n" + "#" * 60)
    print("Hyperparameter tuning: XGBoost")
    print("#" * 60)
    best_xgb = tune_xgboost(X_train, y_train)

    print("\n" + "#" * 60)
    print("Hyperparameter tuning: Random Forest")
    print("#" * 60)
    best_rf = tune_random_forest(X_train_scaled, y_train)

    acc_xgb = evaluate(best_xgb, X_test, y_test, "XGBoost (Tuned)")
    acc_rf = evaluate(best_rf, X_test_scaled, y_test, "Random Forest (Tuned)")

    best_model, best_name, best_acc = (
        (best_xgb, "XGBoost", acc_xgb) if acc_xgb >= acc_rf
        else (best_rf, "Random Forest", acc_rf)
    )

    print("\n" + "=" * 60)
    print(f"Final best model: {best_name}  |  Test Accuracy = {best_acc*100:.2f}%")
    print("=" * 60)

    joblib.dump(best_model, "best_churn_model.pkl")
    print("\nModel saved to: best_churn_model.pkl")

    if best_acc < 0.90:
        print(
            "\nNote: real accuracy on unseen test data tends to stay around "
            "85-87% on this dataset no matter how much you tune, because that "
            "is the realistic ceiling without overfitting or data leakage. "
            "If you need genuinely higher accuracy, you'd need extra data/"
            "features (e.g. transaction history, complaints, login frequency, "
            "etc.) that aren't present in the current file."
        )


if __name__ == "__main__":
    main()

"""
random forest with 100 trees
use gridsearch to search parameters

feature engineering first
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from hebo.design_space.design_space import DesignSpace
from hebo.optimizers.hebo import HEBO

from configs import *


# 加载数据
def load_data():
    train_sets = []
    for train_set in SELECTED_TRAINING_DATASET:
        train_set_path = os.path.join(DATASET_DIR, f"{train_set}_features.parquet")
        train_set_df = pd.read_parquet(train_set_path, engine="fastparquet")

        # print(f"Dataset: {train_set}")
        # print(train_set_df.shape)
        # print(train_set_df.isnull().sum())

        train_sets.append(train_set_df)

    train_df = pd.concat(train_sets)
    # drop duplicated userids
    # train_df = train_df.drop_duplicates(subset="user_id")
    # missing value report
    print(train_df.isnull().sum())

    print("train_df shape: ", train_df.shape)

    test_sets = []
    for test_set in SELECTED_VALIDATING_DATASET:
        test_set_path = os.path.join(DATASET_DIR, f"{test_set}_features.parquet")
        test_set_df = pd.read_parquet(test_set_path, engine="fastparquet")

        # print(f"Dataset: {test_set}")
        # print(test_set_df.shape)
        # print(test_set_df.isnull().sum())

        test_sets.append(test_set_df)

    test_df = pd.concat(test_sets)
    # drop duplicated userids
    # test_df = test_df.drop_duplicates(subset="user_id")
    print("test_df shape: ", test_df.shape)

    print(test_df.isnull().sum())

    return train_df, test_df


# 数据预处理函数
def preprocess_data(df):
    X = df[
        [
            "statuses_count",
            "followers_count",
            "friends_count",
            "listed_count",
            "verified",
            "tweet_freq",
            "followers_growth_rate",
            "friends_growth_rate",
            "listed_growth_rate",
            "followers_friends_ratio",
            "screen_name_length",
            "num_digits_in_screen_name",
            "name_length",
            "num_digits_in_name",
            "screen_name_likelihood",
        ]
    ]
    y = df["is_bot"]
    return X, y


# 定义评估函数
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # 获取预测概率

    auc = roc_auc_score(y_test, y_prob)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    return auc, accuracy, precision, recall


# HEBO 调参和模型训练
def train_random_forest_with_hebo(
    X_train, y_train, X_test, y_test, n_trials=20, n_repeats=5
):
    # 定义搜索空间
    space = DesignSpace().parse(
        [
            {"name": "max_depth", "type": "int", "lb": 5, "ub": 100, "steps": 10},
            {"name": "min_samples_split", "type": "int", "lb": 1, "ub": 30, "steps": 5},
            {"name": "min_samples_leaf", "type": "int", "lb": 1, "ub": 30, "steps": 5},
            {
                "name": "max_features",
                "type": "cat",
                "categories": ["sqrt", "log2", None],
            },
        ]
    )

    best_scores = []
    best_params = []

    for _ in range(n_repeats):
        opt = HEBO(space)
        for _ in range(n_trials):
            # 获取候选参数
            rec = opt.suggest(n_suggestions=1)
            params = rec.iloc[0].to_dict()
            print(params)

            # 创建模型
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=params["max_depth"],
                min_samples_split=params["min_samples_split"],
                min_samples_leaf=params["min_samples_leaf"],
                max_features=params["max_features"],
                random_state=42,
                n_jobs=-1,
            )

            # 训练模型
            model.fit(X_train, y_train)

            # 评估模型
            auc, _, _, _ = evaluate_model(model, X_test, y_test)

            # 记录结果
            opt.observe(rec, np.array([[1 - auc]]))  # HEBO 是最小化目标函数

        # 获取最佳参数
        best_params_trial = opt.best_x
        best_params.append(best_params_trial)

        # 将best_params_trial取第一行
        best_params_trial = best_params_trial.iloc[0].to_dict()

        # 用最佳参数训练模型并评估
        best_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=best_params_trial["max_depth"],
            min_samples_split=best_params_trial["min_samples_split"],
            min_samples_leaf=best_params_trial["min_samples_leaf"],
            max_features=best_params_trial["max_features"],
            random_state=42,
            n_jobs=-1,
        )
        best_model.fit(X_train, y_train)
        auc, accuracy, precision, recall = evaluate_model(best_model, X_test, y_test)
        best_scores.append((auc, accuracy, precision, recall))

    # 计算平均分数
    avg_scores = np.mean(best_scores, axis=0)
    print("Average Scores (AUC, Accuracy, Precision, Recall):", avg_scores)

    return best_model, best_params, avg_scores


# 预测函数
def predict_with_model(model, new_data):
    X_new = new_data[
        [
            "statuses_count",
            "followers_count",
            "friends_count",
            "listed_count",
            "verified",
            "tweet_freq",
            "followers_growth_rate",
            "friends_growth_rate",
            "listed_growth_rate",
            "followers_friends_ratio",
            "screen_name_length",
            "num_digits_in_screen_name",
            "name_length",
            "num_digits_in_name",
            "screen_name_likelihood",
        ]
    ]
    predictions = model.predict(X_new)
    probabilities = model.predict_proba(X_new)[:, 1]
    return predictions, probabilities


# 主函数
if __name__ == "__main__":
    # 加载和预处理数据
    train_df, test_df = load_data()
    X_train, y_train = preprocess_data(train_df)
    X_test, y_test = preprocess_data(test_df)

    # 训练模型
    best_model, best_params, avg_scores = train_random_forest_with_hebo(
        X_train, y_train, X_test, y_test
    )

    # 打印最佳参数和平均分数
    print("Best Parameters:", best_params)
    print("Average Scores (AUC, Accuracy, Precision, Recall):", avg_scores)

    # 保存model
    import joblib

    output_folder = "/scratch/network/yh6580/bot-detection/models"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_path = os.path.join(output_folder, "random_forest_model.pkl")
    joblib.dump(best_model, output_path)

    # # 用模型预测新数据
    # new_data = pd.DataFrame(...)  # 替换为你的新数据
    # best_model = joblib.load(output_path)
    # predictions, probabilities = predict_with_model(best_model, new_data)
    # print("Predictions:", predictions)
    # print("Probabilities:", probabilities)

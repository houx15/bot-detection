import os

import joblib

from datetime import datetime, timedelta

from tqdm import tqdm
import pandas as pd

from utils import log_error

pd.set_option("future.no_silent_downcasting", True)

feature_dir = "/mnt/disk-ccc-covid3-llm/bot-detection"

output_dir = "/mnt/disk-ccc-covid3-llm/bot-detection/result"

topics = ["abortion", "climate-change", "gun", "sexual-orientation"]


model_path = "random_forest_model.pkl"
# 加载模型
model = joblib.load(model_path)


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


def process_topic(topic):
    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 6, 1)
    current_date = start_date

    date_range = [
        current_date + timedelta(days=i) for i in range((end_date - start_date).days)
    ]

    for cur_date in tqdm(date_range, desc=f"Processing {topic}"):
        date_str = cur_date.strftime("%Y-%m-%d")
        data_path = os.path.join(feature_dir, topic, f"{date_str}.parquet")

        if not os.path.exists(data_path):
            continue

        feature = pd.read_parquet(data_path, engine="fastparquet")

        feature["followers_friends_ratio"] = feature["followers_count"] / (
            feature["friends_count"] + 1
        )

        feature["verified"] = feature["verified"].fillna(0)

        feature["verified"] = feature["verified"].astype(int)

        # 如果feature存在na，打印

        if feature.isnull().sum().sum() > 0:
            print(f"Missing values in {date_str}")
            print(feature.shape)
            print(feature.isnull().sum())

        # feature drop na的行
        # feature = feature.dropna()
        # if feature.shape[0] == 0:
        #     continue

        predictions, probabilities = predict_with_model(model, feature)

        # 保留id, predictions, prababilities
        result = feature[["id"]].copy()
        result.loc[:, "predictions"] = predictions
        result.loc[:, "probabilities"] = probabilities

        if not os.path.exists(os.path.join(output_dir, topic)):
            os.makedirs(os.path.join(output_dir, topic))

        output_path = os.path.join(output_dir, topic, f"{date_str}.parquet")
        result.to_parquet(output_path, engine="fastparquet")


def merge(topic):
    topic_prefix = topic.split("-")[0]
    relevance_dir = f"/mnt/disk-ccc-covid3-llm/data-relevance/{topic_prefix}-relevance"

    total_num = 0
    predicted_num = 0
    bot_count = 0

    start_date = datetime(2007, 4, 15)
    end_date = datetime(2023, 6, 1)
    current_date = start_date

    date_range = [
        current_date + timedelta(days=i) for i in range((end_date - start_date).days)
    ]

    for cur_date in tqdm(date_range, desc=f"Merging {topic}"):
        date_str = cur_date.strftime("%Y-%m-%d")
        relevance_path = os.path.join(relevance_dir, f"{date_str}-relevance.pickle4")
        if not os.path.exists(relevance_path):
            continue

        try:
            relevance = pd.read_pickle(relevance_path)
        except:
            relevance = pd.read_parquet(relevance_path, engine="fastparquet")
            relevance = relevance[["relevance"]]

        total_num += relevance.shape[0]

        # relevance的index 是tweet id

        bot_predictions_path = os.path.join(output_dir, topic, f"{date_str}.parquet")
        if not os.path.exists(bot_predictions_path):
            log_error(f"Missing bot predictions for {date_str}")
            continue

        bot_predictions = pd.read_parquet(bot_predictions_path, engine="fastparquet")

        # drop duplicates according to the id column
        bot_predictions = bot_predictions.drop_duplicates("id")

        # # bot_predictions的id的类型
        # print(bot_predictions["id"].dtype)
        # # relevance的index的类型
        # print(relevance.index.dtype)
        # 将bot predictions的id转换为int
        bot_predictions["id"] = bot_predictions["id"].astype(int)

        # merge bot predictions to relevance, using the id to relevance's index, keep relevance's
        relevance = relevance.merge(
            bot_predictions.set_index("id"),
            left_index=True,
            right_index=True,
            how="left",
        )
        # print(relevance)
        # raise Exception

        # prediction不是na的数目为predicted
        predicted_num += relevance["predictions"].notna().sum()

        # bot的数目为1
        bot_count += relevance["predictions"].sum()

        relevance.to_parquet(relevance_path, engine="fastparquet")

    print(f"Total tweets: {total_num}")
    print(f"Predicted tweets: {predicted_num}")
    print(f"Precicted ratio: {predicted_num / total_num}")
    print(f"Bot tweets: {bot_count}")
    print(f"Bot ratio: {bot_count / predicted_num}")


if __name__ == "__main__":
    for topic in topics:
        process_topic(topic)

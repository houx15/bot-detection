"""
统一数据格式为：
user_id,screen_name,name,description,statuses_count,followers_count,friends_count,favourites_count,listed_count,default_profile,profile_use_background_image,verified,user_age,is_bot
存储为parquet
"""

import os

import pandas as pd
import json

from configs import *

from datetime import datetime

from bigram_calculator import calculate_screen_name_likelihood


"""
{
    "created_at": "Fri May 03 20:17:32 +0000 2019", 
    "user": {
        "id": 993674729830612992, 
        "id_str": "993674729830612992", 
        "name": "Connect 4\u20e3", 
        "screen_name": "BotConnectFour", 
        "location": "", 
        "description": "Challenge a friend to disc-dropping fun with a digital game of Connect 4!", 
        "url": "https://t.co/ofTK5VsnFI", 
        "entities": {
            "url": {"urls": [{"url": "https://t.co/ofTK5VsnFI", "expanded_url": "http://howard.bio/bot-connect-4", "display_url": "howard.bio/bot-connect-4", "indices": [0, 23]}]}, 
            "description": {"urls": []}
        }, 
        "protected": false, 
        "followers_count": 11, 
        "friends_count": 1, 
        "listed_count": 1, 
        "created_at": "Tue May 08 02:11:31 +0000 2018", 
        "favourites_count": 67, 
        "utc_offset": null, 
        "time_zone": null, 
        "geo_enabled": false, 
        "verified": false, 
        "statuses_count": 522, 
        "lang": null, 
        "contributors_enabled": false, 
        "is_translator": false, 
        "is_translation_enabled": false, 
        "profile_background_color": "F5F8FA", 
        "profile_background_image_url": null, 
        "profile_background_image_url_https": null, 
        "profile_background_tile": false, 
        "profile_image_url": "http://pbs.twimg.com/profile_images/994376017082048514/oyNhECOm_normal.jpg", 
        "profile_image_url_https": "https://pbs.twimg.com/profile_images/994376017082048514/oyNhECOm_normal.jpg", 
        "profile_banner_url": "https://pbs.twimg.com/profile_banners/993674729830612992/1525751249", 
        "profile_link_color": "1DA1F2", "profile_sidebar_border_color": "C0DEED", "profile_sidebar_fill_color": "DDEEF6", "profile_text_color": "333333", "profile_use_background_image": true, "has_extended_profile": false, "default_profile": true, "default_profile_image": false, "following": false, "follow_request_sent": false, "notifications": false, "translator_type": "none"}}, {"created_at": "Wed Jun 05 10:12:31 +0000 2019", "user": {"id": 971203153269018624, "id_str": "971203153269018624", "name": "fake lyrics from the mountain goats", "screen_name": "faketmglyrics", "location": "Going to...", "description": "(fake) nice boys who make the rocking tunes. I only listen to @mountain_goats, and now I make up my own lyrics. Questions? contact @sheishistoric", "url": "https://t.co/KLRkVFp0Sf", "entities": {"url": {"urls": [{"url": "https://t.co/KLRkVFp0Sf", "expanded_url": "http://mountain-goats.com", "display_url": "mountain-goats.com", "indices": [0, 23]}]}, "description": {"urls": []}}, "protected": false, "followers_count": 55, "friends_count": 1, "listed_count": 1, "created_at": "Wed Mar 07 01:57:30 +0000 2018", "favourites_count": 20, "utc_offset": null, "time_zone": null, "geo_enabled": false, "verified": false, "statuses_count": 34423, "lang": null, "contributors_enabled": false, "is_translator": false, "is_translation_enabled": false, "profile_background_color": "F5F8FA", "profile_background_image_url": null, "profile_background_image_url_https": null, 
        "profile_background_tile": false, 
        "profile_image_url": "http://pbs.twimg.com/profile_images/973359178201772032/J0eDhDcy_normal.jpg", 
        "profile_image_url_https": "https://pbs.twimg.com/profile_images/973359178201772032/J0eDhDcy_normal.jpg", 
        "profile_link_color": "1DA1F2", 
        "profile_sidebar_border_color": "C0DEED", "profile_sidebar_fill_color": "DDEEF6", "profile_text_color": "333333", "profile_use_background_image": true, "has_extended_profile": false, 
        "default_profile": true, 
        "default_profile_image": false, 
        "following": false, "follow_request_sent": false, "notifications": false, 
        "translator_type": "none"}
    }
"""


original_file_mapping = {
    "botwiki": ["botwiki-2019.tsv", "botwiki-2019_tweets.json"],
    "midterm": ["midterm-2018.tsv", "midterm-2018_processed_user_objects.json"],
    "gilani": ["gilani-2017.tsv", "gilani-2017_tweets.json"],
    "cresci-rtbust": ["cresci-rtbust-2019.tsv", "cresci-rtbust-2019_tweets.json"],
    "celebrity": ["celebrity-2019.tsv", "celebrity-2019_tweets.json"],
    "botometer-feedback": [
        "botometer-feedback-2019.tsv",
        "botometer-feedback-2019_tweets.json",
    ],
    "political-bots": ["political-bots-2019.tsv", "political-bots-2019_tweets.json"],
}

time_format = "%a %b %d %H:%M:%S %z %Y"
time_format_midterm = "%a %b %d %H:%M:%S %Y"


def process_tsv_and_json(data_name):
    print(f"Processing {data_name}")

    if data_name not in original_file_mapping:
        print(f"{data_name} not in original_file_mapping")
        return

    tsv_file_name, json_file_name = original_file_mapping[data_name]

    tsv_file_path = os.path.join(DATASET_DIR, data_name, tsv_file_name)
    json_file_path = os.path.join(DATASET_DIR, data_name, json_file_name)

    df = pd.read_csv(tsv_file_path, sep="\t", header=None)
    df.columns = ["user_id", "is_bot"]

    # 如果is_bot是human/bot，转化为0-1
    df["is_bot"] = df["is_bot"].map(
        {"human": 0, "bot": 1, "bot (automated)": 1, "cyborg": 1}
    )

    # 将user_id转为str设置为index
    df["user_id"] = df["user_id"].astype(str)
    df.set_index("user_id", inplace=True)
    # 将df转化为dict, 示例: {"123123123": 0, "1235325432": 1}
    bot_dict = df["is_bot"].to_dict()

    with open(json_file_path, "r") as f:
        user_objects = json.load(f)

    all_data = []

    for single_tweet_info in user_objects:
        try:
            assert "user" in single_tweet_info or "user_id" in single_tweet_info
        except AssertionError:
            print(single_tweet_info)
            raise AssertionError

        single_user_info = {}

        if "user" in single_tweet_info:
            user_id = str(single_tweet_info["user"]["id"])
            user_obj = single_tweet_info["user"]

            tweet_time = datetime.strptime(single_tweet_info["created_at"], time_format)
            user_time = datetime.strptime(user_obj["created_at"], time_format)
        else:
            user_id = str(single_tweet_info["user_id"])
            user_obj = single_tweet_info

            tweet_time = datetime.strptime(
                single_tweet_info["probe_timestamp"], time_format_midterm
            )
            user_time = datetime.strptime(
                user_obj["user_created_at"], time_format_midterm
            )

        user_age = (tweet_time - user_time).total_seconds() / 3600

        if user_id in bot_dict:
            try:
                single_user_info = {
                    "user_id": user_id,
                    "screen_name": user_obj["screen_name"],
                    "name": user_obj["name"],
                    "description": user_obj["description"],
                    "statuses_count": user_obj["statuses_count"],
                    "followers_count": user_obj["followers_count"],
                    "friends_count": user_obj["friends_count"],
                    "favourites_count": user_obj["favourites_count"],
                    "listed_count": user_obj["listed_count"],
                    "default_profile": user_obj["default_profile"],
                    "profile_use_background_image": user_obj[
                        "profile_use_background_image"
                    ],
                    "verified": user_obj["verified"],
                    "user_age": user_age,
                    "is_bot": bot_dict[user_id],
                }
                all_data.append(single_user_info)
            except KeyError as e:
                print(f"KeyError: {e}")
                print(single_tweet_info)
                raise e

    organized_data = pd.DataFrame(all_data)

    output_path = os.path.join(DATASET_DIR, f"{data_name}.parquet")
    organized_data.to_parquet(output_path, engine="fastparquet", index=False)


def process_all_tsvs():
    for data_name in SELECTED_TRAINING_DATASET + SELECTED_VALIDATING_DATASET:
        process_tsv_and_json(data_name)


def process_cresci():
    data_folder = "/scratch/network/yh6580/bot-detection/data/cresci/datasets_full.csv"
    folders = [
        "genuine_accounts.csv",
        "fake_followers.csv",
        "social_spambots_1.csv",
        "social_spambots_2.csv",
        "social_spambots_3.csv",
        "traditional_spambots_1.csv",
        "traditional_spambots_2.csv",
        "traditional_spambots_3.csv",
        "traditional_spambots_4.csv",
    ]
    true_accounts = ["genuine_accounts.csv"]

    all_df = []

    for folder in folders:
        print(f"Processing {folder}")

        user_data = pd.read_csv(os.path.join(data_folder, folder, "users.csv"))

        """
        "id","name","screen_name","statuses_count","followers_count","friends_count","favourites_count","listed_count","url","lang","time_zone","location","default_profile","default_profile_image","geo_enabled","profile_image_url","profile_banner_url","profile_use_background_image","profile_background_image_url_https","profile_text_color","profile_image_url_https","profile_sidebar_border_color","profile_background_tile","profile_sidebar_fill_color","profile_background_image_url","profile_background_color","profile_link_color","utc_offset","is_translator","follow_request_sent","protected","verified","notifications","description","contributors_enabled","following","created_at","timestamp","crawled_at","updated","test_set_1","test_set_2"
        """

        if folder == "traditional_spambots_1.csv":
            user_data["created_at"] = user_data["created_at"].apply(
                lambda x: (
                    int(str(x).rstrip("L"))
                    if isinstance(x, (int, str)) and str(x).endswith("L")
                    else x
                )
            )
            user_data["created_at"] = pd.to_datetime(
                user_data["created_at"], unit="ms", utc=True
            )
        else:
            user_data["created_at"] = pd.to_datetime(
                user_data["created_at"], format="%a %b %d %H:%M:%S %z %Y", utc=True
            )
        if "crawled_at" in user_data:
            user_data["probe_time"] = pd.to_datetime(
                user_data["crawled_at"], format="%Y-%m-%d %H:%M:%S", utc=True
            )
        else:
            user_data["probe_time"] = pd.to_datetime(
                user_data["updated"], format="%Y-%m-%d %H:%M:%S", utc=True
            )

        # 计算小时差
        user_data["user_age"] = (
            user_data["probe_time"] - user_data["created_at"]
        ).dt.total_seconds() / 3600

        # verified 用0填充缺失值
        user_data["verified"].fillna(0, inplace=True)

        # print(user_data["verified"].value_counts())
        # print(user_data["verified"].isnull().sum())

        # # 有多少个friends_count是0
        # print(user_data["friends_count"].value_counts())
        # print(user_data["friends_count"].isnull().sum())

        user_data = user_data[
            [
                "id",
                "screen_name",
                "name",
                "description",
                "statuses_count",
                "followers_count",
                "friends_count",
                "favourites_count",
                "listed_count",
                "default_profile",
                "profile_use_background_image",
                "verified",
                "user_age",
            ]
        ]

        if folder in true_accounts:
            user_data["is_bot"] = 0
        else:
            user_data["is_bot"] = 1

        all_df.append(user_data)

    all_df = pd.concat(all_df)
    all_df.to_parquet(
        os.path.join(DATASET_DIR, "cresci.parquet"), engine="fastparquet", index=False
    )


def feature_preparation():
    for dataset in SELECTED_TRAINING_DATASET + SELECTED_VALIDATING_DATASET:
        df = pd.read_parquet(os.path.join(DATASET_DIR, f"{dataset}.parquet"))

        df.fillna({"name": "", "screen_name": ""}, inplace=True)

        df["tweet_freq"] = df["statuses_count"] / df["user_age"]
        df["followers_growth_rate"] = df["followers_count"] / df["user_age"]
        df["friends_growth_rate"] = df["friends_count"] / df["user_age"]
        df["listed_growth_rate"] = df["listed_count"] / df["user_age"]
        df["followers_friends_ratio"] = df["followers_count"] / (
            df["friends_count"] + 1
        )
        df["screen_name_length"] = df["screen_name"].apply(lambda x: len(x))
        df["num_digits_in_screen_name"] = df["screen_name"].apply(
            lambda x: sum(c.isdigit() for c in x)
        )
        df["name_length"] = df["name"].apply(lambda x: len(x))
        df["num_digits_in_name"] = df["name"].apply(
            lambda x: sum(c.isdigit() for c in x)
        )
        df["screen_name_likelihood"] = df["screen_name"].apply(
            calculate_screen_name_likelihood
        )

        df = df[
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
                "is_bot",
            ]
        ]

        df.to_parquet(
            os.path.join(DATASET_DIR, f"{dataset}_features.parquet"),
            engine="fastparquet",
            index=False,
        )


if __name__ == "__main__":
    # process_cresci()
    feature_preparation()

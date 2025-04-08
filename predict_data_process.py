import os
import json
import ijson

import gc

import tarfile
import gzip
import pickle
import time

import pandas as pd
from datetime import datetime, timedelta

from bigram_calculator import calculate_screen_name_likelihood

from tqdm import tqdm

from utils import log_error

from configs import *


data_dir = ""

targz_dir = ""

base_dir = BASE_DIR

profile_dir = PROFILE_DIR

profile_curation_dir = ""

feature_dir = os.path.join(base_dir, "user_features")

topics = ["abortion", "climate-change", "gun", "sexual-orientation"]
# topics = ["climate-change", "gun", "sexual-orientation"]


text_dir = ""
relevance_dir = ""

text_folder_mapping = {
    "gun": "gun-opinion",
    "climate": "climate-change-opinion",
    "abortion": "abortion-opinion",
    "sexual": "sexual-orientation-opinion",
}


def process_single_file(tweet_df, users_df):
    # merge users_df to tweet_df
    users_df = users_df.set_index("id")
    users_df.rename(
        columns={
            "created_at": "user_created_at",
        },
        inplace=True,
    )

    for col in users_df.columns:
        tweet_df.loc[:, col] = tweet_df["author_id"].map(users_df[col])

    # missings num
    print(tweet_df.isnull().sum())
    print(tweet_df.shape)
    print(tweet_df, users_df)
    print(tweet_df["author_id"].dtype, users_df.index.dtype)
    print(users_df.columns)
    raise ValueError("stop")

    tweet_df["statuses_count"] = tweet_df["public_metrics.tweet_count"]
    tweet_df["followers_count"] = tweet_df["public_metrics.followers_count"]
    tweet_df["friends_count"] = tweet_df["public_metrics.following_count"]
    tweet_df["listed_count"] = tweet_df["public_metrics.listed_count"]

    tweet_df["created_at"] = pd.to_datetime(tweet_df["created_at"], utc=True)
    tweet_df["user_created_at"] = pd.to_datetime(tweet_df["user_created_at"], utc=True)

    tweet_df["user_age"] = (
        tweet_df["created_at"] - tweet_df["user_created_at"]
    ).dt.total_seconds() / 3600

    tweet_df.rename(columns={"username": "screen_name"}, inplace=True)

    # fillna name
    tweet_df.fillna({"name": "", "screen_name": ""}, inplace=True)

    # 替换为0的user_age
    tweet_df.loc[tweet_df["user_age"] == 0, "user_age"] = 0.1

    tweet_df["tweet_freq"] = tweet_df["statuses_count"] / tweet_df["user_age"]
    tweet_df["followers_growth_rate"] = (
        tweet_df["followers_count"] / tweet_df["user_age"]
    )
    tweet_df["friends_growth_rate"] = tweet_df["friends_count"] / tweet_df["user_age"]
    tweet_df["listed_growth_rate"] = tweet_df["listed_count"] / tweet_df["user_age"]
    tweet_df["followers_friends_ratio"] = tweet_df["followers_count"] / (
        tweet_df["friends_count"] + 1
    )
    tweet_df["screen_name_length"] = tweet_df["screen_name"].apply(
        lambda x: len(str(x))
    )
    tweet_df["num_digits_in_screen_name"] = tweet_df["screen_name"].apply(
        lambda x: sum(c.isdigit() for c in str(x))
    )
    tweet_df["name_length"] = tweet_df["name"].apply(lambda x: len(str(x)))
    tweet_df["num_digits_in_name"] = tweet_df["name"].apply(
        lambda x: sum(c.isdigit() for c in str(x))
    )
    tweet_df["screen_name_likelihood"] = tweet_df["screen_name"].apply(
        calculate_screen_name_likelihood
    )

    tweet_df = tweet_df[
        [
            "id",
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

    tweet_df["verified"] = tweet_df["verified"].astype(int)
    tweet_df["verified"].fillna(0, inplace=True)

    return tweet_df


def process_topic_feature_extraction(topic):
    # 如果data_dir不存在，从targz_dir解压数据
    if not os.path.exists(os.path.join(feature_dir, topic)):
        os.makedirs(os.path.join(feature_dir, topic))

    if not os.path.exists(os.path.join(data_dir, topic)):
        print(f"Extracting data for {topic}")
        # os.system(f"tar -xzf {targz_dir}/{topic}.tar.gz -C {data_dir}")
        print(f"Data extracted for {topic}")

    start_date = datetime(2023, 1, 1)
    end_date = datetime(2023, 6, 1)

    current_date = start_date

    date_range = [
        current_date + timedelta(days=i) for i in range((end_date - start_date).days)
    ]

    for cur_date in tqdm(date_range, desc=f"Processing {topic}"):

        date_str = cur_date.strftime("%Y-%m-%d")

        cur_tweet_data = []
        cur_user_data = []

        files = [
            f
            for f in os.listdir(os.path.join(data_dir, topic))
            if f.startswith(date_str) and f.endswith(".ipickle4")
        ]

        for file in files:
            file_path = os.path.join(data_dir, topic, file)

            # try:
            ipickle_data = pd.read_pickle(file_path)
            # print(ipickle_data)
            users_df = ipickle_data["users"]
            print(users_df.index.unique().shape[0])
            cur_user_data.append(users_df)

            if "tweets" in ipickle_data:
                tweet_df = ipickle_data["tweets"]
                print(tweet_df.index.unique().shape[0])
            else:
                tweet_path = file_path.replace(".ipickle4", ".pickle4")
                tweet_df = pd.read_pickle(tweet_path)

            tweet_df = tweet_df[["id", "created_at", "author_id"]].copy()

            cur_tweet_data.append(tweet_df)

        users_df = pd.concat(cur_user_data)
        tweet_df = pd.concat(cur_tweet_data)

        # 打印有users里面有多少个独立的index
        print(users_df.index.unique().shape[0])
        print(tweet_df["author_id"].unique().shape[0])

        # users_df drop duplicate index
        users_df = users_df[~users_df.index.duplicated(keep="first")]

        features = process_single_file(users_df=users_df, tweet_df=tweet_df)
        # cur_date_data.append(features)

        # except Exception as e:
        #     print(f"Error processing {file}: {e}")
        #     log_error(f"Error processing {file}: {e}")

        # if len(cur_date_data) > 0:
        #     cur_date_data = pd.concat(cur_date_data)
        #     cur_date_data.to_parquet(
        #         os.path.join(feature_dir, topic, f"{date_str}.parquet")
        #     )

    print(f"Feature extraction completed for {topic}")
    # delete the extracted tar gz data
    # os.system(f"rm -rf {os.path.join(data_dir, topic)}")


def get_all_relevant_user_ids():
    """
    只处理relevant的部分
    """
    unique_user_set = set()
    topics = ["gun", "climate", "abortion", "sexual"]
    start_date = datetime(2006, 1, 1)
    end_date = datetime(2023, 6, 1)

    current_date = start_date

    date_range = [
        current_date + timedelta(days=i) for i in range((end_date - start_date).days)
    ]

    for topic in topics:
        topic_text_folder = os.path.join(text_dir, text_folder_mapping[topic])
        topic_relevance_folder = os.path.join(relevance_dir, f"{topic}-relevance")

        for date in tqdm(date_range, desc=f"processing {topic}"):
            date_str = date.strftime("%Y-%m-%d")
            text_file = os.path.join(topic_text_folder, f"{date_str}-text.pickle4")
            if not os.path.exists(text_file):
                continue

            relevance_file = os.path.join(
                topic_relevance_folder, f"{date_str}-relevance.pickle4"
            )
            if not os.path.exists(relevance_file):
                continue

            data = pd.read_pickle(text_file)
            try:
                relevance = pd.read_pickle(relevance_file)
            except:
                relevance = pd.read_parquet(relevance_file, engine="fastparquet")
                relevance = relevance[["relevance"]]

            relevance_ids = relevance[relevance["relevance"] == 1].index
            data = data.loc[relevance_ids]

            # 将data中的author id添加到set中
            unique_user_set.update(data["author_id"].unique())

    print(f"author id num: {len(unique_user_set)}")

    output_path = os.path.join(base_dir, "author_ids.json")
    with open(output_path, "w") as f:
        json.dump(list(unique_user_set), f)


def generate_single_tar_file_index(tar_path, index_file_path):
    """
    生成 tar.gz 中实际存在文件的索引
    :param tar_path: tar.gz 文件路径
    :param index_file_path: 索引文件保存路径
    """
    with tarfile.open(tar_path, "r:gz") as tar:
        with open(index_file_path, "w") as index_file:
            for member in tar.getmembers():
                # 只记录文件名
                index_file.write(member.name + "\n")


def get_all_tar_file_index():
    """
    遍历某个文件夹下的所有 tar.gz 文件，并生成对应的索引文件
    :param tar_dir: 存放 tar.gz 文件的目录
    :param index_dir: 索引文件保存目录
    """
    tar_dir = profile_dir
    index_dir = os.path.join(profile_curation_dir, "index")
    os.makedirs(index_dir, exist_ok=True)

    for file_name in os.listdir(tar_dir):
        if file_name.endswith(".tar.gz"):
            tar_path = os.path.join(tar_dir, file_name)
            index_file_path = os.path.join(index_dir, f"{file_name}.index")

            # 跳过已存在的索引文件
            if os.path.exists(index_file_path):
                print(f"Index file already exists, skipping: {index_file_path}")
                continue

            generate_single_tar_file_index(tar_path, index_file_path)


def convert_time_format(input_time: str) -> str:
    """
    将时间从格式 'Fri May 25 13:18:07 +0000 2018' 转换为 '2006-07-15T08:41:38.000Z'

    :param input_time: 输入的时间字符串
    :return: 转换后的时间字符串
    """
    # 定义输入时间格式
    input_format = "%a %b %d %H:%M:%S %z %Y"
    # 定义目标时间格式
    output_format = "%Y-%m-%dT%H:%M:%S.%fZ"

    # 解析输入时间字符串
    parsed_time = datetime.strptime(input_time, input_format)
    # 转换为目标格式
    output_time = parsed_time.strftime(output_format)

    return output_time


profile_zip_files = [
    "v2-2023feb",
    "v2-2022dec",
    "v2-2022oct",
    "v2-2022aug",
    "v2-2022jun",
    "v2-2022apr",
    "v2-2021oct",
    "v2-2021dec",
    "v1-2020aug",
]


class ExtractUserInfo(object):
    def __init__(self):
        self.matched_users = self.load_processed_user_ids()
        self.user_ids = set()
        self.load_unique_user_ids()

        self.save_threshold = 50000  # 每50000个保存一次
        self.finished_zip_files = self.load_finished_zip_files()

        self.user_infos = []
        self.save_count = 0

    def extract_from_file(self, user_id, file_obj, mtime):
        """

        v2
        data-twitter-profile-v2-2023feb/1000-profile.pickle4

        {'username': 'percep2al', 'name': 'Jordy Mont-Reynaud', 'public_metrics': {'followers_count': 6051, 'following_count': 225, 'tweet_count': 97, 'listed_count': 43}, 'id': '1000', 'profile_image_url': 'https://pbs.twimg.com/profile_images/1511727588045209601/QAToVCGo_normal.jpg', 'verified': False, 'created_at': '2006-07-15T08:41:38.000Z'}
        """
        try:
            profile = pickle.load(file_obj)
            self.user_infos.append(
                {
                    "id": user_id,
                    "name": profile["name"],
                    "screen_name": profile["username"],
                    "created_at": profile["created_at"],
                    "followers_count": profile["public_metrics"]["followers_count"],
                    "friends_count": profile["public_metrics"]["following_count"],
                    "listed_count": profile["public_metrics"]["listed_count"],
                    "statuses_count": profile["public_metrics"]["tweet_count"],
                    "verified": profile["verified"],
                    "crawled_at": mtime,
                }
            )

            self.matched_users.add(user_id)
            # pop out from user_ids
            self.user_ids.remove(user_id)

            if len(self.user_infos) >= self.save_threshold:
                self.save_user_info()

            return "success"
        except Exception as e:
            print(f"Error processing {user_id}: {e}")
            log_error(f"Error processing {user_id}: {e}")
            return "fail"

    def extract_from_file_v1(self, user_id, file_obj, mtime):
        """

        v1
        data-profile-curation/home/junming/virus/data-twitter-profile-v1-2020aug/userid-1000003075590025217.json.gz

        {'id': 1000003075590025217, 'id_str': '1000003075590025217', 'name': 'jah kirae 👨\u200d❤️\u200d💋\u200d👨', 'screen_name': 'CardiB_Romania', 'location': 'The fiery pits of Romania', 'description': 'mad? hurt? aww... well, have a good day! bye! | backup acc @jah_TheSequel | not affiliated with any celebrities', 'url': None, 'entities': {'description': {'urls': []}}, 'protected': False, 'followers_count': 635, 'friends_count': 524, 'listed_count': 5, 'created_at': 'Fri May 25 13:18:07 +0000 2018', 'favourites_count': 66471, 'utc_offset': None, 'time_zone': None, 'geo_enabled': False, 'verified': False, 'statuses_count': 13545, 'lang': None, 'status': {'created_at': 'Wed Aug 26 20:18:05 +0000 2020', 'id': 1298716368821350406, 'id_str': '1298716368821350406', 'full_text': '@BODAKRICCH WAP promo gc?', 'truncated': False, 'display_text_range': [12, 25], 'entities': {'hashtags': [], 'symbols': [], 'user_mentions': [{'screen_name': 'BODAKRICCH', 'name': 'deja🤍 | #BLM', 'id': 1173386806051713027, 'id_str': '1173386806051713027', 'indices': [0, 11]}], 'urls': []}, 'source': '<a href="http://twitter.com/download/android" rel="nofollow">Twitter for Android</a>', 'in_reply_to_status_id': 1298716005867241478, 'in_reply_to_status_id_str': '1298716005867241478', 'in_reply_to_user_id': 1173386806051713027, 'in_reply_to_user_id_str': '1173386806051713027', 'in_reply_to_screen_name': 'BODAKRICCH', 'geo': None, 'coordinates': None, 'place': None, 'contributors': None, 'is_quote_status': False, 'retweet_count': 0, 'favorite_count': 0, 'favorited': False, 'retweeted': False, 'lang': 'en'}, 'contributors_enabled': False, 'is_translator': False, 'is_translation_enabled': False, 'profile_background_color': '000000', 'profile_background_image_url': 'http://abs.twimg.com/images/themes/theme1/bg.png', 'profile_background_image_url_https': 'https://abs.twimg.com/images/themes/theme1/bg.png', 'profile_background_tile': False, 'profile_image_url': 'http://pbs.twimg.com/profile_images/1291601019349217285/ytNxDnJ3_normal.jpg', 'profile_image_url_https': 'https://pbs.twimg.com/profile_images/1291601019349217285/ytNxDnJ3_normal.jpg', 'profile_banner_url': 'https://pbs.twimg.com/profile_banners/1000003075590025217/1596776656', 'profile_link_color': '6700B3', 'profile_sidebar_border_color': '000000', 'profile_sidebar_fill_color': '000000', 'profile_text_color': '000000', 'profile_use_background_image': False, 'has_extended_profile': True, 'default_profile': False, 'default_profile_image': False, 'following': False, 'follow_request_sent': False, 'notifications': False, 'translator_type': 'none'}
        """
        try:
            with gzip.open(file_obj, "rt", encoding="utf-8") as f:
                profile = json.load(f)
            self.user_infos.append(
                {
                    "id": user_id,
                    "name": profile["name"],
                    "screen_name": profile["screen_name"],
                    "created_at": convert_time_format(profile["created_at"]),
                    "followers_count": profile["followers_count"],
                    "friends_count": profile["friends_count"],
                    "listed_count": profile["listed_count"],
                    "statuses_count": profile["statuses_count"],
                    "verified": profile["verified"],
                    "crawled_at": mtime,
                }
            )

            self.matched_users.add(user_id)
            # pop out from user_ids
            self.user_ids.remove(user_id)

            if len(self.user_infos) >= self.save_threshold:
                self.save_user_info()

            return "success"
        except Exception as e:
            print(f"Error processing {user_id}: {e}")
            log_error(f"Error processing {user_id}: {e}")
            return "fail"

    def format_info_to_feature(self, df):
        df["created_at"] = pd.to_datetime(
            df["created_at"], format="%Y-%m-%dT%H:%M:%S.%fZ"
        )
        df["crawled_at"] = pd.to_datetime(df["crawled_at"], unit="s")
        df["user_age"] = (df["crawled_at"] - df["created_at"]).dt.total_seconds() / 3600
        df.fillna({"name": "", "screen_name": ""}, inplace=True)
        df.loc[df["user_age"] == 0, "user_age"] = 0.1
        df["tweet_freq"] = df["statuses_count"] / df["user_age"]
        df["followers_growth_rate"] = df["followers_count"] / df["user_age"]
        df["friends_growth_rate"] = df["friends_count"] / df["user_age"]
        df["listed_growth_rate"] = df["listed_count"] / df["user_age"]
        df["followers_friends_ratio"] = df["followers_count"] / (
            df["friends_count"] + 1
        )
        df["screen_name_length"] = df["screen_name"].apply(lambda x: len(str(x)))
        df["num_digits_in_screen_name"] = df["screen_name"].apply(
            lambda x: sum(c.isdigit() for c in str(x))
        )
        df["name_length"] = df["name"].apply(lambda x: len(str(x)))
        df["num_digits_in_name"] = df["name"].apply(
            lambda x: sum(c.isdigit() for c in str(x))
        )
        df["screen_name_likelihood"] = df["screen_name"].apply(
            calculate_screen_name_likelihood
        )
        df["verified"] = df["verified"].astype(int)
        df["verified"].fillna(0, inplace=True)
        df = df[
            [
                "id",
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
        return df

    def save_user_info(self):
        feature_df = pd.DataFrame(self.user_infos)
        feature_df = self.format_info_to_feature(feature_df)
        output_path = os.path.join(
            feature_dir, f"user-{int(time.time())}-{self.save_count}.parquet"
        )
        feature_df.to_parquet(output_path, engine="fastparquet", index=False)
        self.save_count += 1
        self.user_infos = []
        self.save_processed_user_ids()

        gc.collect()

    def load_finished_zip_files(self):
        """
        加载已经处理过的zip文件
        :return:
        """
        if not os.path.exists(os.path.join(base_dir, "finished_zip_files.json")):
            return []

        with open(os.path.join(base_dir, "finished_zip_files.json"), "r") as f:
            finished_zip_files = json.load(f)

        return finished_zip_files

    def load_unique_user_ids(self):
        """
        加载唯一的用户id
        :return:
        """
        with open(os.path.join(base_dir, "author_ids.json"), "r") as f:
            for user_id in ijson.items(f, "item"):
                if user_id not in self.matched_users:
                    self.user_ids.add(user_id)

    def load_processed_user_ids(self):
        """
        加载已经处理过的用户id
        :return:
        """
        if not os.path.exists(os.path.join(base_dir, "processed_user_ids.json")):
            return set()
        processed_ids = []
        with open(os.path.join(base_dir, "processed_user_ids.json"), "r") as f:
            processed_ids = json.load(f)

        return set(processed_ids)

    def save_processed_user_ids(self):
        """
        保存已经处理过的用户id
        :return:
        """
        with open(os.path.join(base_dir, "processed_user_ids.json"), "w") as f:
            json.dump(list(self.matched_users), f)
        print(f"Saved processed user ids: {len(self.matched_users)}")

    def save_finished_zip_files(self):
        """
        保存已经处理过的zip文件
        :return:
        """
        with open(os.path.join(base_dir, "finished_zip_files.json"), "w") as f:
            json.dump(self.finished_zip_files, f)
        print(f"Saved finished zip files: {len(self.finished_zip_files)}")

    def process(self):
        for single_tar_file in profile_zip_files:
            if single_tar_file in self.finished_zip_files:
                print(f"File {single_tar_file} already processed, skipping.")
                continue
            tar_path = os.path.join(
                profile_dir, f"data-twitter-profile-{single_tar_file}.tar.gz"
            )

            i = 0
            new_user = 0
            target_start = 0

            if single_tar_file == "v2-2023feb":
                target_start = 33400000

            with tarfile.open(tar_path, "r:gz") as tar:
                member = tar.next()
                while member and i < target_start:
                    i += 1
                    if i % 100000 == 0:
                        print(f"Skipped {i} files")
                        gc.collect()
                    member = tar.next()
                # for member in tqdm(tar, desc=f"Processing {single_tar_file}"):
                while member:
                    i += 1
                    if i % 100000 == 0:
                        print(
                            f"Processed {i} files in {single_tar_file}, added {new_user} new users"
                        )
                        gc.collect()
                    if "v1" in single_tar_file:
                        process_flag = member.isfile() and member.name.endswith(
                            ".json.gz"
                        )
                    else:
                        process_flag = member.isfile() and member.name.endswith(
                            ".pickle4"
                        )
                    if process_flag:
                        if "v1" in single_tar_file:
                            user_id = (
                                member.name.split("/")[-1].split("-")[1].split(".")[0]
                            )
                        else:
                            user_id = member.name.split("/")[-1].split("-")[0]
                        if user_id not in self.user_ids:
                            member = tar.next()
                            continue
                        if user_id in self.matched_users:
                            member = tar.next()
                            continue
                        file_obj = tar.extractfile(member)
                        if file_obj is None:
                            member = tar.next()
                            continue
                        with file_obj:
                            # 读取文件的修改时间
                            mtime = member.mtime
                            processed = (
                                self.extract_from_file_v1(user_id, file_obj, mtime)
                                if "v1" in single_tar_file
                                else self.extract_from_file(user_id, file_obj, mtime)
                            )
                            if processed == "success":
                                new_user += 1
                            del file_obj

                    # gc.collect()

                    member = tar.next()

            if len(self.user_infos) > 0:
                self.save_user_info()

            self.finished_zip_files.append(single_tar_file)
            self.save_finished_zip_files()


if __name__ == "__main__":
    extractor = ExtractUserInfo()
    print("inited")
    extractor.process()

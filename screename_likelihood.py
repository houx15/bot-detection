import os
import pandas as pd
from datetime import datetime, timedelta


def get_2m_users():
    # 初始化变量
    start_year = 2023  # 从 2023 年开始
    end_year = 2013  # 向前遍历到 2013 年
    unique_usernames = set()  # 存储唯一的 screen names
    data_dir = "/mnt/disk-ccc-covid3-llm/data-tweet-curation/abortion"  # 数据文件夹路径
    target_count = 2000000  # 目标是 2M unique screen names

    # 遍历年份
    for year in range(start_year, end_year - 1, -1):  # 从 2023 到 2013，逐年递减
        print(f"Processing year: {year}")

        # 遍历每一天
        current_date = datetime(year, 1, 1)
        while current_date.year == year:
            date_str = current_date.strftime("%Y-%m-%d")

            # 获取当天的所有文件
            files = [
                f
                for f in os.listdir(data_dir)
                if f.startswith(date_str) and f.endswith(".ipickle4")
            ]

            # 遍历当天的所有文件
            for file in files:
                file_path = os.path.join(data_dir, file)

                # 加载 ipickle 文件
                try:
                    ipickle_data = pd.read_pickle(file_path)
                    users_df = ipickle_data["users"]  # 提取 users 表

                    # 提取 username 并添加到 set 中
                    unique_usernames.update(users_df["username"].dropna().unique())

                    # 如果已经达到目标数量，停止程序
                    if len(unique_usernames) >= target_count:
                        break
                except Exception as e:
                    print(f"Error loading {file}: {e}")

            # 如果已经达到目标数量，停止程序
            if len(unique_usernames) >= target_count:
                break

            # 日期向后移动一天
            current_date += timedelta(days=1)

        # 如果已经达到目标数量，停止程序
        if len(unique_usernames) >= target_count:
            break

    # 保存结果
    output_path = "unique_usernames_2M.txt"
    with open(output_path, "w") as f:
        for username in unique_usernames:
            f.write(f"{username}\n")

    print(
        f"Collected {len(unique_usernames)} unique usernames. Saved to {output_path}."
    )


from collections import defaultdict
import json


def cacluate_bigram_probabilities():
    # 假设你已经有了 2M 的 screen names
    with open("unique_usernames_2M.txt") as f:
        screen_names = [line.strip() for line in f]

    # 1. 初始化 bigram 计数器
    bigram_counts = defaultdict(int)
    total_bigrams = 0

    # 2. 遍历所有 screen names，统计 bigram 频率
    for name in screen_names:
        for i in range(len(name) - 1):  # 提取 bigram
            bigram = name[i : i + 2]
            bigram_counts[bigram] += 1
            total_bigrams += 1

    # 3. 计算每个 bigram 的概率分布
    bigram_probabilities = {
        bigram: count / total_bigrams for bigram, count in bigram_counts.items()
    }

    # 4. 保存 bigram 概率分布到文件
    bigram_output_path = "bigram_probabilities.json"
    with open(bigram_output_path, "w") as f:
        json.dump(bigram_probabilities, f)

    for name, likelihood in list(bigram_probabilities.items())[:15]:
        print(f"{name}: {likelihood}")

    print(
        f"Saved {len(bigram_probabilities)} bigram probabilities to {bigram_output_path}"
    )


import numpy as np

# 1. 加载 bigram 概率分布
with open("bigram_probabilities.json") as f:
    bigram_probabilities = json.load(f)


# 2. 定义计算 screen_name 概率的函数
def calculate_screen_name_likelihood(name, bigram_probabilities):
    bigram_probs = []
    for i in range(len(name) - 1):
        bigram = name[i : i + 2]
        # 如果某个 bigram 没有出现过，使用一个很小的默认概率
        prob = bigram_probabilities.get(bigram, 1e-10)
        bigram_probs.append(prob)

    # 几何平均值
    if bigram_probs:  # 确保 bigram_probs 不为空
        geometric_mean = np.prod(bigram_probs) ** (1 / len(bigram_probs))
    else:
        geometric_mean = 0  # 如果 name 长度小于 2

    return geometric_mean

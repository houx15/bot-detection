import numpy as np
import json

# 1. 加载 bigram 概率分布
with open("bigram_probabilities.json") as f:
    bigram_probabilities = json.load(f)


# 2. 定义计算 screen_name 概率的函数
def calculate_screen_name_likelihood(name):
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

import re
import json
import os
from bert_score import score

Max = 0

root_dir = r"C:\Users\Trist\Desktop\campus\pythonProject\SportsSum\SportsSum\sports_data"
pair_path = "pair.txt"
important_path = "important.txt"
index_path = os.listdir(root_dir)

for index in index_path:
    pair_file_path = os.path.join(root_dir, index, pair_path)
    pair_file = open(pair_file_path, "r", encoding="utf8")

    important_file_path = os.path.join(root_dir, index, important_path)
    important_file = open(important_file_path, "a+", encoding="utf8")

    lines = pair_file.readlines()

    for line in lines:
        flag = line.split("&", 1)
        live_pair = []
        news_pair = []
        live_pair.append(flag[0])
        news_pair.append(flag[1])
        P, R, F1 = score(live_pair, news_pair, lang="zh", verbose=True)
        sc = float(F1)
        if sc >= 0.7:
            important_file.write("###" + str(flag[0]) + str(flag[1]))
        else:
            important_file.write("???" + str(flag[0]) + str(flag[1]))
        print(f"System level F1 score: {F1.mean():.3f}")

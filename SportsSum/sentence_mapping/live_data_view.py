import json
import os
import pandas as pd
root_dir = r"C:\Users\Trist\Desktop\campus\pythonProject\SportsSum\SportsSum\sports_data"
live_dir = "live.json"
text_path = os.listdir(root_dir)
for file_path in text_path:
    txt_path = os.path.join(root_dir, file_path, "live.txt")
    file = open(txt_path, "a+", encoding="utf8")
    path = os.path.join(root_dir, file_path, live_dir)
    json_file = open(path, "r")
    item_list1 = json.load(json_file)
    item_list = item_list1["result"]["data"]
    for list1 in item_list:
        list1.pop("id")
        list1.pop("s1")
        list1.pop("s2")
        list1.pop("s")
    item_list.reverse()
    sentence_frame = pd.DataFrame(item_list)
    sentence_list = list(sentence_frame.iterrows())
    for index, row in sentence_frame.iterrows():
        str = row['t'] + row['m'] + "\n"
        print(file_path + "   " + str)
        file.write(str)
print("OK")
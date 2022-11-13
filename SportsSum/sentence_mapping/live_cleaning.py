import os
import re

root_dir = r"C:\Users\Trist\Desktop\campus\pythonProject\SportsSum\SportsSum\sports_data"
live_dir = "live.txt"
live_clean_path = "live_cleaning.txt"
text_path = os.listdir(root_dir)

for file_path in text_path:
    txt_clean_path = os.path.join(root_dir, file_path, live_clean_path)
    txt_clean_file = open(txt_clean_path, "a+", encoding="utf8")
    live_path = os.path.join(root_dir, file_path, live_dir)
    live_file = open(live_path, "r", encoding="utf8")
    live_lines = live_file.readlines()
    for line in live_lines:
        string = "\d+'"
        pattern = re.compile(string)
        flag = bool(re.search(pattern, line))
        if (flag):
            txt_clean_file.write(line)
            print(file_path)
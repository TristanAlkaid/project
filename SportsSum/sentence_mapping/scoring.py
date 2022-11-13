import os

root_dir = r"C:\Users\Trist\Desktop\campus\pythonProject\SportsSum\SportsSum\sports_data"
pair_dir = "pair.txt"
pair_live_dir = "pair_live.txt"
pair_news_dir = "pair_news.txt"
text_path = os.listdir(root_dir)

for file_path in text_path:
    pair_path = os.path.join(root_dir, file_path, pair_dir)
    pair_file = open(pair_path, "r", encoding="utf8")
    pair_live_path = os.path.join(root_dir, file_path, pair_live_dir)
    pair_live_file = open(pair_live_path, "a+", encoding="utf8")

    pair_news_path = os.path.join(root_dir, file_path, pair_news_dir)
    pair_news_file = open(pair_news_path, "a+", encoding="utf8")

    pair_lines = pair_file.readlines()
    for line in pair_lines:
        lines = line.split("&", 1)
        print(file_path + lines[0] + lines[1])
        pair_live_file.write(lines[0])
        pair_live_file.write(lines[1])

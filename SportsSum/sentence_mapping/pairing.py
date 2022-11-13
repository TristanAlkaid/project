import os

root_dir = r"C:\Users\Trist\Desktop\campus\pythonProject\SportsSum\SportsSum\sports_data"
live_row = "live_row.txt"
news_row = "news_row.txt"
pairing_path = "pair.txt"
text_path = os.listdir(root_dir)

for file_path in text_path:
    live_row_path = os.path.join(root_dir, file_path, live_row)
    live_row_file = open(live_row_path, "r", encoding="utf8")

    news_row_path = os.path.join(root_dir, file_path, news_row)
    news_row_file = open(news_row_path, "r", encoding="utf8")

    pair_path = os.path.join(root_dir, file_path, pairing_path)
    pair_file = open(pair_path, "a+", encoding="utf8")

    live_row_lines = live_row_file.readlines()
    news_row_lines = news_row_file.readlines()

    for news_row_line in news_row_lines:
        news_resul_flag = news_row_line.split("'", 1)
        news_time = int(news_resul_flag[0])
        for live_row_line in live_row_lines:
            live_line_flag = live_row_line.split("'", 1)
            live_time = int(live_line_flag[0])
            if live_time >= news_time and live_time <= news_time + 3:
                a = live_row_line.split("\n")
                b = news_row_line.split("\n")
                pairing = a[0] + "&" + b[0] + "\n"
                print(pairing)
                pair_file.write(pairing)
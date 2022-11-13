import os

root_dir = r"C:\Users\Trist\Desktop\campus\pythonProject\SportsSum\SportsSum\sports_data"
live_clean = "live_cleaning.txt"
live_clean_row = "live_row.txt"
news_result = "news_result.txt"
news_result_row = "news_row.txt"
text_path = os.listdir(root_dir)

for file_path in text_path:
    live_clean_path = os.path.join(root_dir, file_path, live_clean)
    live_clean_file = open(live_clean_path, "r", encoding="utf8")
    live_clean_row_path = os.path.join(root_dir, file_path, live_clean_row)
    live_clean_row_file = open(live_clean_row_path, "a+", encoding="utf8")

    news_result_path = os.path.join(root_dir, file_path, news_result)
    news_result_file = open(news_result_path, "r", encoding="utf8")
    news_result_row_path = os.path.join(root_dir, file_path, news_result_row)
    news_result_row_file = open(news_result_row_path, "a+", encoding="utf8")

    live_lines = live_clean_file.readlines()
    news_result_lines = news_result_file.readlines()

    temp = 0
    for live_line in live_lines:
        live_line_flag = live_line.split("'", 1)
        if (temp <= int(live_line_flag[0])):
            temp = int(live_line_flag[0])
            live_clean_row_file.write(str(live_line_flag[0]) + "'" + str(live_line_flag[1]))
            print("live " + file_path + " " + live_line_flag[0] + "'" + live_line_flag[1])
        else:
            a = int(live_line_flag[0])
            a = a + temp
            live_clean_row_file.write(str(a) + "'" + live_line_flag[1])
            print("live " + file_path + " " + str(a) + "'" + live_line_flag[1])

    temp = 0
    for news_result_line in news_result_lines:
        news_result_line_flag = news_result_line.split("'", 1)
        if (temp <= int(news_result_line_flag[0])):
            temp = int(news_result_line_flag[0])
            news_result_row_file.write(news_result_line_flag[0] + "'" + news_result_line_flag[1])
            print("news " + file_path + " " + news_result_line_flag[0] + "'" + news_result_line_flag[1])
        else:
            b = int(news_result_line_flag[0])
            a = b + 45
            news_result_row_file.write(str(a) + "'" + news_result_line_flag[1])
            print("news " + file_path + " " + str(a) + "'" + news_result_line_flag[1])

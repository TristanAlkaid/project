import os
import re


def find(path):
    file = open(path, encoding="utf8")
    mystr = file.read()
    pattern = re.compile('[。！][^。!]*一开场[^。！]*[。！]|[。！][^。！]*开场后[^。！]*[。!]|[。!][^。！]*开场[\d]+分钟[^。！]*[。！]|[。！][^。！]*开始[\d]+分钟[^。！][^。！]*第[\d]+分钟[^。！]*[。！]|[。！][^。！]*[\d]+分钟[^。！]*[。！]')
    extract = re.findall(pattern, mystr)
    return extract


def write(path, str):
    with open(path, "a+", encoding="utf8") as f:
        f.write(str)


if __name__ == '__main__':
    root_dir = r"C:\Users\Trist\Desktop\campus\pythonProject\SportsSum\SportsSum\sports_data"
    news_result_dir = "news_result.txt"
    news_dir = "news.txt"
    path_index = os.listdir(root_dir)
    for path_list in path_index:
        news_path = os.path.join(root_dir, path_list, news_dir)
        news_result_path = os.path.join(root_dir, path_list, news_result_dir)
        news_result_list = find(news_path)
        pattern_1 = re.compile('一开场|开场后')
        pattern_2 = re.compile('。|<strong>|</strong>|\n|！')
        for sentence in news_result_list:
            new_sentence = re.sub(pattern_1, '第1分钟', sentence)
            new_sentence = re.sub(pattern_2, '', new_sentence)
            pattern_3 = re.compile('[1-9]+\.?[0-9]*')
            time_all = re.findall(pattern_3, new_sentence)
            news_result = str(time_all[0]) + "'" + new_sentence + "\n"
            print(path_list +  "   " +str(time_all[0]) + "'" + new_sentence)
            write(news_result_path, news_result)
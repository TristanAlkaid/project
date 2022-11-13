import os
import json

txt_path = r"C:\Users\Trist\Desktop\campus\pythonProject\SportsSum\SportsSum\sports_data\csl_0598\live.txt"
json_path = r"C:\Users\Trist\Desktop\campus\pythonProject\SportsSum\SportsSum\sports_data\csl_0598\live1.json"

with open(txt_path, 'r', encoding='utf-8') as f:  # 打开txt文件
    txt = f.read()
    with open(json_path, 'a', encoding='utf-8') as file:  # 创建一个json文件，mode设置为'a'
        json.dump(txt, file, ensure_ascii=False)  # 将字典d写入json文件中，并设置ensure_ascii = False,主要是因为汉字是ascii字符码,若不指定该参数，那么文字格式就会是ascii码
        file.write('\n')

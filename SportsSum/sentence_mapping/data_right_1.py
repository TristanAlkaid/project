import  json

path = r"C:\Users\Trist\Desktop\campus\pythonProject\SportsSum\SportsSum\sports_data\csl_0598\live.json"
path2 = r"C:\Users\Trist\Desktop\campus\pythonProject\SportsSum\SportsSum\sports_data\csl_0598\live.txt"
json_file = open(path, "r")
item_list1 = json.load(json_file)
file = open(path2, "a+", encoding="utf8")
file.write(str(item_list1))
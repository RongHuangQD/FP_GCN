import os

project_dir = os.path.dirname(__file__)
datasets_dir = os.path.join(project_dir, 'poi.data')
all_poi_path = os.path.join(datasets_dir, 'allPOIs_2411.txt')
place2vec_path = os.path.join(datasets_dir, 'place2vec.txt')
graphcontent_path = os.path.join(datasets_dir, 'poi.data.graphcontent.txt')

# 读取text1文件中的数据
text1_data = []
with open(all_poi_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if line:
            fid, type2, lon, lat, zoneID = line.split(',')
            text1_data.append((fid, type2, lon, lat, zoneID))

# 读取text2文件中的数据
text2_data = {}
with open(place2vec_path, 'r', encoding='utf-8') as file:
    for line in file:
        line = line.strip()
        if line:
            parts = line.split(' ')
            type2 = parts[0]
            vector = ' '.join(parts[1:])
            text2_data[type2] = vector

# 匹配数据并生成输出
output_data = []
for fid, type2, lon, lat, zoneID in text1_data:
    if type2 in text2_data:
        vector = text2_data[type2]
        output_data.append(f"{fid}\t{vector} {type2}\t{zoneID}")

# 打印输出
for item in output_data:
    print(item)

with open(graphcontent_path, 'w', encoding='utf-8') as file:
    file.write('\n'.join(output_data))
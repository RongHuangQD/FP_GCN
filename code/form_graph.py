import os
import numpy as np
from scipy.spatial import distance_matrix

project_dir = os.path.dirname(__file__)
datasets_dir = os.path.join(project_dir, '../data', 'poi.data')
def formGraph(zone, K=16):
    dist_mat = distance_matrix(zone, zone)
    poiCount = len(zone)
    adjlists = [[] for _ in range(poiCount)]

    for i in range(poiCount):
        dist = list(dist_mat[i])
        for _ in range(K):
            minIndex = dist.index(min(dist))
            dist[minIndex] = float('inf')
            adjlists[i].append(minIndex)

    return adjlists

all_poi_path = os.path.join(datasets_dir, f'allPOIs_2411.txt')
zoneCount = 2411

with open(all_poi_path, 'r', encoding='utf-8') as f:
    f.readline()
    zone_dic_wordname = [[] for _ in range(zoneCount)]
    zone_dic_wordcordinate = [[] for _ in range(zoneCount)]
    zone_dic_id = [[] for _ in range(zoneCount)]

    while True:
        temp = f.readline().strip('\n')
        if not temp:
            break
        temp = temp.split(',')
        zone_dic_wordname[int(temp[4])].append(temp[1])
        zone_dic_wordcordinate[int(temp[4])].append([float(temp[2]), float(temp[3])])
        zone_dic_id[int(temp[4])].append(int(temp[0]))

for i in range(zoneCount):
    if len(zone_dic_wordcordinate[i]) < 3:
        continue
    zone = np.array(zone_dic_wordcordinate[i])
    G = formGraph(zone)
    edges = []
    for source_node, target_nodes in enumerate(G):
        for target_node in target_nodes:
            edges.append((source_node, target_node))

    zone_dic_id_values = zone_dic_id[i]
    id_to_value = {idx: value for idx, value in enumerate(zone_dic_id_values)}

    edges_with_values = []
    for edge_pair in edges:
        if edge_pair[0] in id_to_value and edge_pair[1] in id_to_value:
            edges_with_values.append((id_to_value[edge_pair[0]], id_to_value[edge_pair[1]]))

    output_file_path = os.path.join(datasets_dir, f'poi.data.poi_graph_zone{[i]}.txt')
    with open(output_file_path, "w") as file:
        for edge_pair in edges_with_values:
            file.write(f"{edge_pair[0]}\t{edge_pair[1]}\n")
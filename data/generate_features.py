import gensim
from scipy import spatial
import os

def main():
    project_dir = os.path.dirname(__file__)
    datasets_dir = os.path.join(project_dir, 'poi.data')
    all_poi_path = os.path.join(datasets_dir, 'allPOIs_2411.txt')
    corups_path = os.path.join(datasets_dir, 'corups_place2vec.txt')
    zone_count = 2411
    neighbor_num = 22
    vec_dim = 200

    def read_allPOIs_to_zonePOIs(path):
        f = open(path, encoding='utf-8')  # 打开 all_POI
        f.readline()  # 读出表头信息

        zone_dic_wordname = {}
        zone_dic_wordcordinate = {}
        for i in range(0, zone_count):  # 初始化
            zone_dic_wordname[i] = []
            zone_dic_wordcordinate[i] = []
        while True:
            temp = f.readline().strip('\n')
            if not temp:
                break
            temp = temp.split(',')
            zone_dic_wordname[eval(temp[4])].append(temp[1])
            zone_dic_wordcordinate[eval(temp[4])].append([eval(temp[2]), eval(temp[3])])
        return zone_dic_wordname, zone_dic_wordcordinate

    def generate_corups_place2vec(zone_dic_wordname, zone_dic_wordcordinate, path, K=4):
        with open(path, "w", encoding="utf-8") as f:
            zoneCount = len(zone_dic_wordname)
            for i in range(0, zoneCount):  # 对每一个区域
                zone_wordname = zone_dic_wordname[i]
                poiCount = len(zone_wordname)
                if poiCount == 0:
                    continue
                if poiCount < K + 2:
                    for n in range(0, poiCount):
                        f.write(zone_wordname[n] + ' ')
                    f.write('\n')
                    continue
                dist_mat = spatial.distance_matrix(zone_dic_wordcordinate[i], zone_dic_wordcordinate[i])
                dist_mat = dist_mat.tolist()
                for j in range(0, poiCount):  # 对区域中的每一个POI找出最近的K个邻居
                    strNN = ''
                    dist = dist_mat[j]
                    minIndex = dist.index(min(dist))
                    dist[minIndex] = 10000
                    for m in range(0, K):
                        minIndex = dist.index(min(dist))
                        dist[minIndex] = 10000
                        strNN += zone_wordname[minIndex] + ' '
                        if m == K // 2 - 1:  # 中间位置跟K有关
                            strNN += zone_wordname[j] + ' '
                    f.write(strNN + '\n')

    def train_model(corups_path, datasets_dir):
        with open(corups_path, 'r', encoding="utf-8") as f:
            sentences = []
            for line in f:
                cols = line.strip().split(' ')
                sentences.append(cols)
        model = gensim.models.Word2Vec(sentences, sg=1, vector_size=vec_dim, alpha=0.025,
                                        window=neighbor_num + 1, min_count=1, sample=1e-3, seed=1,
                                        workers=4, min_alpha=0.0001, hs=0, negative=20, cbow_mean=1)

        # save
        place2vec_path = os.path.join(datasets_dir, "place2vec.txt")
        model.wv.save_word2vec_format(place2vec_path, binary=False)

        # Remove the first line from the saved word vectors file
        with open(place2vec_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        with open(place2vec_path, 'w', encoding='utf-8') as f:
            f.writelines(lines[1:])  # Write all lines except the first one
        model.save(os.path.join(datasets_dir, "place2vec.model"))

        return model

    # 1、读取所有的POI
    zone_dic_wordname, zone_dic_wordcordinate = read_allPOIs_to_zonePOIs(all_poi_path)

    # 2、生成语料库
    generate_corups_place2vec(zone_dic_wordname, zone_dic_wordcordinate, corups_path, neighbor_num)

    # 3、训练模型
    model = train_model(corups_path, datasets_dir)

if __name__ == "__main__":
    main()
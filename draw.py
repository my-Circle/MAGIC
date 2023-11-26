import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def get_fer_data(data_path="data_embed_npy6.npy",
                 label_path="label_npu6.npy"):
    """
	该函数读取上一步保存的两个npy文件，返回data和label数据
    Args:
        data_path:
        label_path:

    Returns:
        data: 样本特征数据，shape=(BS,embed)
        label: 样本标签数据，shape=(BS,)
        n_samples :样本个数
        n_features：样本的特征维度

    """
    data = np.load(data_path)
    label = np.load(label_path)
    n_samples, n_features = data.shape

    return data, label, n_samples, n_features


# color_map = ['r', 'y', 'k', 'g', 'b', 'm', 'c']  # 7个类，准备7种颜色
# color_map = ['r', 'y',  'g', 'b', 'm', 'c',
#              'lightcoral', 'orange', 'springgreen', 'deepskyblue', 'royalblue', 'violet',
#              'coral', 'olive', 'lime', 'cyan', 'midnightblue', 'darkviolet',
#              'sienna', 'gold', 'deeppink', 'slategrey']
color_map = ['r', 'springgreen', 'deepskyblue', 'royalblue', 'violet', 'orange',
             'lightcoral', 'y',  'g', 'b', 'm', 'c',
             'coral', 'olive', 'lime', 'cyan', 'midnightblue', 'darkviolet',
             'deeppink', 'slategrey', 'sienna', 'gold']  #换个颜色顺序

def plot_embedding_2D(data, label, title):
    """

    """
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)
    fig = plt.figure()
    for i in range(data.shape[0]):
        if label[i] == 0:
            continue
        elif label[i] == 101:
            label[i] = 0
        elif label[i] == 102:
            label[i] = 1
        elif label[i] == 1996:
            label[i] = 2
        elif label[i] == 1997:
            label[i] = 3
        elif label[i] == 1037:
            label[i] = 4
        elif label[i] == 2000:
            label[i] = 5
        elif label[i] == 1998:
            label[i] = 6
        elif label[i] == 2001:
            label[i] = 7
        elif label[i] == 2003:
            label[i] = 8
        elif label[i] == 1999:
            label[i] = 9
        elif label[i] == 2002:
            label[i] = 10
        elif label[i] == 2022:
            label[i] = 11
        elif label[i] == 2009:
            label[i] = 12
        elif label[i] == 2029:
            label[i] = 13
        elif label[i] == 2007:
            label[i] = 14
        elif label[i] == 2030:
            label[i] = 15
        elif label[i] == 8036:
            label[i] = 16
        elif label[i] == 2006:
            label[i] = 17
        elif label[i] == 2008:
            label[i] = 18
        elif label[i] == 2019:
            label[i] = 19
        elif label[i] == 2004:
            label[i] = 20
        else:
            label[i] = 21  #continue 看聚类效果选是否赋颜色
        plt.plot(data[i, 0], data[i, 1], marker='o', markersize=1, color=color_map[label[i]])
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    return fig


def main():
    data, label, n_samples, n_features = get_fer_data()  # 根据自己的路径合理更改

    print('Begining......')

    # 调用t-SNE对高维的data进行降维，得到的2维的result_2D，shape=(samples,2)
    tsne_2D = TSNE(n_components=2, init='pca', random_state=0)
    result_2D = tsne_2D.fit_transform(data)

    print('Finished......')
    fig1 = plot_embedding_2D(result_2D, label, 't-SNE')  # 将二维数据用plt绘制出来
    fig1.show()
    plt.pause(50)


if __name__ == '__main__':
    main()


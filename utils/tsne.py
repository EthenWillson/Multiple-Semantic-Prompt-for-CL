import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def tSNE(x, y, t, d=None):
    plt.clf()

    tsne = TSNE(random_state=105)
    tsne.fit_transform(x.cpu())  # 进行数据降维

    L1 = [x[0] for x in tsne.embedding_]
    L2 = [x[1] for x in tsne.embedding_]

    plt.scatter(L1,L2,c=y.cpu(),marker='x')

    
    if d is not None:
        tsne_d = TSNE(random_state=105,perplexity=1)
        tsne_d.fit_transform(d.cpu())
        d1 = [x[0] for x in tsne_d.embedding_]
        d2 = [x[1] for x in tsne_d.embedding_]
        plt.scatter(d1,d2,s=100,c='r',marker='*')
    plt.savefig('./img/pic_task{}.png'.format(t))

    return tsne

def tSNE_all(x, t, d=None):
    plt.clf()

    tsne = TSNE(random_state=105)

    i = 0
    for x_i in x:
        tsne.fit_transform(x_i.cpu())  # 进行数据降维

        L1 = [x_i[0] for x_i in tsne.embedding_]
        L2 = [x_i[1] for x_i in tsne.embedding_]

        plt.scatter(L1,L2,c=[i for j in range(x_i.shape[0])],marker='x')
        i += 1

    
    if d is not None:
        tsne_d = TSNE(random_state=105,perplexity=1)
        tsne_d.fit_transform(d.cpu())
        d1 = [x[0] for x in tsne_d.embedding_]
        d2 = [x[1] for x in tsne_d.embedding_]
        plt.scatter(d1,d2,s=100,c='r',marker='*')
    plt.savefig('./img/pic_task{}.png'.format(t))

    return tsne

    # emb_data = pd.DataFrame(tsne.embedding_, index = x.index)  # 转换数据格式

    # import matplotlib.pyplot as plt
    # plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    # plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

    # # 不同类别用不同颜色和样式绘图
    # for k, color in enumerate(['r.','go','b^']):
    #     d = emb_data[r['聚类类别'] == k]
    #     plt.plot(d[0], d[1], color, label=k)

    # plt.legend()
    # plt.show()

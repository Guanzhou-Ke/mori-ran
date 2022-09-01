"""
Visualization result.
------

Author: Guanzhou Ke.
Email: guanzhouk@gmail.com
Date: 2022/08/14
"""
from mimetypes import init
import torch
import numpy as np
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE


def plot_scatter(ax, x, labels=None, legend=None):
    s = 16
    if labels is not None:
        y = np.unique(labels)
        for i in y:
            idx = labels == i
            ax.scatter(x[idx, 0], x[idx, 1], s=s, label=f"{i}")
    else:
        ax.scatter(x[:, 0], x[:, 1], s=s, label=legend)

def vis_hidden(history_path):
    his = torch.load(history_path)
    tsne = TSNE(n_components=2, perplexity=30,  random_state=166)
    labels = his['labels']
    labels = labels.numpy()
    _, axes = plt.subplots(2, 5, figsize=(20, 8))
    for ii, epoch in enumerate([0, 5, 10, 15, 95]):
        hs = his[f'hidden_{epoch}']['hs']
        h1, h2 = hs['h1'][0], hs['h2'][0]
        h1, h2 = h1.numpy(), h2.numpy()
        sh1, sh2 = tsne.fit_transform(h1), tsne.fit_transform(h2)
        ax = axes[0][ii]
        plot_scatter(ax, sh1, legend='view 1')
        plot_scatter(ax, sh2, legend='view 2')
        if ii == 0:
            ax.legend()
        z = his[f'hidden_{epoch}']['z']
        sz = tsne.fit_transform(z)
        ax = axes[1][ii]
        plot_scatter(ax, sz, labels)
        if ii == 0:
            ax.legend()
    plt.savefig('./experiments/results/vis.pdf', format='pdf')
            

def calc_stats(history_path, task):
    his = torch.load(history_path)
    if task == 'clustering':
        acc = [] 
        nmi = []
        ari = []
        p = []
        fscore = []
        for seed in his['seeds']:
            acc.append(np.max(his[f'seed:{seed}']['acc']))
            nmi.append(np.max(his[f'seed:{seed}']['nmi']))
            ari.append(np.max(his[f'seed:{seed}']['ari']))
            p.append(np.max(his[f'seed:{seed}']['p']))
            fscore.append(np.max(his[f'seed:{seed}']['fscore']))
        print(f"ACC: {np.mean(acc):.4f} (std: {np.std(acc):.4f}), NMI: {np.mean(nmi):.4f} (std: {np.std(nmi):.4f}), ARI: {np.mean(ari):.4f} (std: {np.std(ari):.4f}) p: {np.mean(p):.4f}, fscore: {np.mean(fscore):.4f}")
    else:
        acc = []
        p = []
        fscore = []
        for seed in his['seeds']:
            acc.append(np.max(his[f'seed:{seed}']['acc']))
            p.append(np.max(his[f'seed:{seed}']['P']))
            fscore.append(np.max(his[f'seed:{seed}']['f_score']))
        print(f"ACC: {np.mean(acc):.4f} (std: {np.std(acc):.4f}), p: {np.mean(p):.4f} (std: {np.std(p):.4f}), fscore: {np.mean(fscore):.4f} (std: {np.std(fscore):.4f})")

if __name__ == '__main__':
    # model = 'conan'
    model = 'moriran'
    eid = 0
    ablation = False
    # ttype = 'clustering'
    # for ds in ['Scene-15', 'LandUse-21', 'Caltech101-20', 'NoisyMNIST']:
    #     try:
    #         print(ds)
    #         if ablation:
    #             path = f'./experiments/results/ablation/{model}-{eid}/{ds}/{ttype}/history.log'
    #         else:
    #             path = f'./experiments/results/{model}-{eid}/{ds}/{ttype}/history.log'
    #         calc_stats(path, ttype)
    #     except:
    #         continue
    hidden_path = './experiments/results/hidden.data'
    vis_hidden(hidden_path)
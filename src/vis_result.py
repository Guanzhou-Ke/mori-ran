"""
Visualization result.
------

Author: Guanzhou Ke.
Email: guanzhouk@gmail.com
Date: 2022/08/14
"""
import torch
import numpy as np
            

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
    ttype = 'clustering'
    for ds in ['Scene-15', 'LandUse-21', 'Caltech101-20', 'NoisyMNIST']:
        try:
            print(ds)
            if ablation:
                path = f'./experiments/results/ablation/{model}-{eid}/{ds}/{ttype}/history.log'
            else:
                path = f'./experiments/results/{model}-{eid}/{ds}/{ttype}/history.log'
            calc_stats(path, ttype)
        except:
            continue

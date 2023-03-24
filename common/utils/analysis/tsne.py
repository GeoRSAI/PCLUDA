"""
@author: Junguang Jiang
@contact: JiangJunguang1123@outlook.com
"""
import torch
import matplotlib

matplotlib.use('Agg')
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as col
import seaborn as sns

def visualize(source_feature: torch.Tensor, target_feature: torch.Tensor,
              filename: str, source_color='blue', target_color='red'):
    """
    Visualize features from different domains using t-SNE.

    Args:
        source_feature (tensor): features from source domain in shape :math:`(minibatch, F)`
        target_feature (tensor): features from target domain in shape :math:`(minibatch, F)`
        filename (str): the file name to save t-SNE
        source_color (str): the color of the source features. Default: 'r'
        target_color (str): the color of the target features. Default: 'b'

    """
    sns.set_theme(style="white")

    source_feature = source_feature.numpy()
    target_feature = target_feature.numpy()
    features = np.concatenate([source_feature, target_feature], axis=0)

    # map features to 2-d using TSNE
    X_tsne = TSNE(n_components=2, random_state=33).fit_transform(features)

    # domain labels, 1 represents source while 0 represents target
    domains = np.concatenate((np.ones(len(source_feature)), np.zeros(len(target_feature))))

    # visualize using matplotlib
    fig, ax = plt.subplots(figsize=(10, 10))

    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([target_color, source_color]), s=20)
    # sns.scatterplot(X_tsne[:, 0], X_tsne[:, 1], c=domains, cmap=col.ListedColormap([target_color, source_color]), s=20)
    # plt.legend()


    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()

    plt.savefig(filename, dpi=600)

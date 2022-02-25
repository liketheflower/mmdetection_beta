import pandas as pd
import numpy as np
from PIL import Image as im
import itertools

import os
import seaborn as sns
import matplotlib.pylab as plt

# network = "swin"
def get_featuremap(network):
    # network = "resnet"
    if network == "swin":
        outputs = pd.read_pickle(
            "/Users/jimmy/repos/jimmy/output_featuremap/swin_outs.pkl"
        )
    else:
        outputs = pd.read_pickle(
            "/Users/jimmy/repos/jimmy/output_featuremap/resnet_outs.pkl"
        )
    outs = list(outputs["outs"])

    last_layer = outs[-1]
    # the shape is [1, 768, 25, 35]
    last_np = last_layer.cpu().detach().numpy()
    print(last_np.shape)
    last_np = np.squeeze(last_np, axis=0)
    return last_np


swin = get_featuremap("swin")
resnet = get_featuremap("resnet")
# save_path = "./feature_map_vis/"
save_path = "./feature_map_visi_separate/"
os.makedirs(save_path, exist_ok=True)


def vis_ij(i, j):
    # for i, j in itertools.combinations(range(swin.shape[0]), 2):
    plt.close()
    plt.subplot(2, 2, 1)
    ax = sns.heatmap(resnet[i])
    plt.title("ResNet-50 feature map " + str(i))
    plt.subplot(2, 2, 2)
    ax = sns.heatmap(swin[i])
    plt.title("Swin-T feature map " + str(i))
    plt.subplot(2, 2, 3)
    ax = sns.heatmap(resnet[j])
    plt.title("ResNet-50 feature map " + str(j))
    plt.subplot(2, 2, 4)
    ax = sns.heatmap(swin[j])
    plt.title("Swin-T feature map " + str(j))
    fig = plt.gcf()
    fig.set_size_inches((11.97, 8.36), forward=False)
    plt.tight_layout()
    plt.savefig(save_path + str(i).zfill(3) + "_" + str(j).zfill(3) + ".pdf", dpi=200)
    plt.close()


def vis():
    for i in range(swin.shape[0]):
        plt.close()
        ax = sns.heatmap(resnet[i])
        plt.title("ResNet-50 feature map " + str(i))
        plt.savefig(save_path + "resnet_" + str(i).zfill(3) + ".pdf", dpi=300)
        plt.close()
        ax = sns.heatmap(swin[i])
        plt.title("Swin-T feature map " + str(i))
        plt.savefig(save_path + "swin_" + str(i).zfill(3) + ".pdf", dpi=300)


# vis()
vis_ij(36, 111)

import pandas as pd
import numpy as np
from PIL import Image as im

import seaborn as sns
import matplotlib.pylab as plt

#network = "swin"
network = "resnet"
save_np_file = True
save_rm_extreme_hist = True


def normalize(a, min_=None, max_=None, clip=False):
    if min_ is None or max_ is None:
        min_ = np.min(a)
        max_ = np.max(a)
    new_a = (a - min_) / (max_ - min_)
    if clip:
        new_a = np.clip(new_a, 0, 1)
    return new_a


if network == "swin":
    outputs = pd.read_pickle("/Users/jimmy/repos/jimmy/output_featuremap/swin_outs.pkl")
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

print(last_np.shape)

"""
resnet

swin

>>> 25*32
800
>>> 35*24
840
"""
C, H, W = last_np.shape
print(np.max(last_np))
print(np.min(last_np))

last_np_normalize = normalize(last_np)
# last_np_normalize = normalize(last_np_normalize, 0.57, 0.7)
# last_np_normalize = last_np_normalize*255
# last_np_normalize = last_np_normalize.dtype(np.uint8)
# output_img = output_img.dtype(np.uint8)
idx = 0
if network == "swin":
    output_img = np.ones((H * 32, W * 24), dtype=np.float)
    for i in range(32):
        for j in range(24):
            output_img[i * H : i * H + H, j * W : j * W + W] = last_np_normalize[idx]
            idx += 1
else:
    output_img = np.ones((H * 64, W * 32), dtype=np.float)
    for i in range(64):
        for j in range(32):
            output_img[i * H : i * H + H, j * W : j * W + W] = last_np_normalize[idx]
            idx += 1


last_np_normalize_flat = last_np_normalize.ravel()
if save_np_file:
    np.save(network + "last_np_normalize_flat.npy", last_np_normalize_flat)
plt.hist(last_np_normalize_flat, bins=200)
plt.show()
plt.close()
# uniform_data = np.random.rand(10, 12)

if save_np_file:
    np.save(network + "output_img_with_extreme_val_new.npy", output_img)
ax = sns.heatmap(output_img)
plt.show()
# plt.savefig("swin_heap_300.pdf", dpi=300)
# remove outlier
if network == "swin":
    last_np_normalize = normalize(last_np_normalize, 0.57, 0.7, clip=True)
else:
    last_np_normalize = normalize(last_np_normalize, 0, 0.1, clip=True)

last_np_normalize_flat = last_np_normalize.ravel()
if save_rm_extreme_hist:
    np.save(network + "last_np_normalize_flat_no_extreme.npy", last_np_normalize_flat)
plt.hist(last_np_normalize_flat, bins=200)
plt.show()
# output_img = output_img.dtype(np.uint8)
idx = 0
if network == "swin":
    output_img = np.ones((H * 32, W * 24), dtype=np.float)
    for i in range(32):
        for j in range(24):
            output_img[i * H : i * H + H, j * W : j * W + W] = last_np_normalize[idx]
            idx += 1
else:
    output_img = np.ones((H * 64, W * 32), dtype=np.float)
    for i in range(64):
        for j in range(32):
            output_img[i * H : i * H + H, j * W : j * W + W] = last_np_normalize[idx]
            idx += 1

if save_np_file:
    np.save(network + "output_img_without_extreme_val_new.npy", output_img)
last_np_normalize_flat = last_np_normalize.ravel()
plt.close()
plt.hist(last_np_normalize_flat, bins=200)
plt.show()
plt.close()
# uniform_data = np.random.rand(10, 12)
ax = sns.heatmap(output_img)
plt.show()

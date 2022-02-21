import pandas as pd
import numpy as np
from PIL import Image as im

import seaborn as sns
import matplotlib.pylab as plt


def normalize(a, min_=None, max_=None, clip=False):
    if min_ is None or max_ is None:
        min_ = np.min(a)
        max_ = np.max(a)
    new_a = (a - min_) / (max_ - min_)
    if clip:
        new_a = np.clip(new_a, 0, 1)
    return new_a


outputs = pd.read_pickle("/Users/jimmy/repos/jimmy/output_featuremap/swin_outs.pkl")
outs = list(outputs["outs"])

last_layer = outs[-1]
# the shape is [1, 768, 25, 35]
last_np = last_layer.cpu().detach().numpy()
print(last_np.shape)
last_np = np.squeeze(last_np, axis=0)

print(last_np.shape)
"""
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
output_img = np.ones((H * 32, W * 24), dtype=np.float)
# output_img = output_img.dtype(np.uint8)
idx = 0
for i in range(32):
    for j in range(24):
        output_img[i * H : i * H + H, j * W : j * W + W] = last_np_normalize[idx]
        idx += 1


last_np_normalize_flat = last_np_normalize.ravel()
plt.hist(last_np_normalize_flat, bins=200)
plt.show()
plt.close()
# uniform_data = np.random.rand(10, 12)
ax = sns.heatmap(output_img)
plt.show()
# plt.savefig("swin_heap.pdf", dpi=500)
# remove outlier
last_np_normalize = normalize(last_np_normalize, 0.57, 0.7, clip=True)

output_img = np.ones((H * 32, W * 24), dtype=np.float)
# output_img = output_img.dtype(np.uint8)
idx = 0
for i in range(32):
    for j in range(24):
        output_img[i * H : i * H + H, j * W : j * W + W] = last_np_normalize[idx]
        idx += 1


last_np_normalize_flat = last_np_normalize.ravel()
plt.hist(last_np_normalize_flat, bins=200)
plt.show()
plt.close()
# uniform_data = np.random.rand(10, 12)
ax = sns.heatmap(output_img)
plt.show()

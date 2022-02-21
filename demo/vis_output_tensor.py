import pandas as pd
import numpy as np
from PIL import Image as im


def normalize(a):
    min_ = np.min(a)
    max_ = np.max(a)
    new_a = (a - min_) / (max_ - min_)
    return new_a

outputs = pd.read_pickle("swin_outs.pkl")
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
#last_np_normalize = last_np_normalize*255
#last_np_normalize = last_np_normalize.dtype(np.uint8)
output_img = np.ones((H*32, W*24), dtype=np.float)
#output_img = output_img.dtype(np.uint8)
idx = 0
for i in range(32):
    for j in range(24):
        output_img[i*H:i*H+H, j*W:j*W +W] = last_np[idx]
        idx += 1


import seaborn as sns
import matplotlib.pylab as plt

#uniform_data = np.random.rand(10, 12)
ax = sns.heatmap(output_img, linewidth=0.5)
plt.savefig("swin_heap.pdf", dpi=500)

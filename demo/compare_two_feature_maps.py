"""
Compare the output feature maps
resnetlast_np_normalize_flat.npy		resnetoutput_img_without_extreme_val.npy	swinoutput_img_with_extreme_val.npy
resnetoutput_img_with_extreme_val.npy		swinlast_np_normalize_flat.npy			swinoutput_img_without_extreme_val.npy

"""
import pandas as pd
import numpy as np
from PIL import Image as im

import seaborn as sns
import matplotlib.pylab as plt

# Feature map without remove outlier
conv_color = "#66bd63"
trans_color = "#f46d43"

swin_img = np.load("swinoutput_img_with_extreme_val.npy")
resnet_img = np.load("resnetoutput_img_with_extreme_val.npy")
swin_img_no_extreme = np.load("swinoutput_img_without_extreme_val.npy")
resnet_img_no_extreme = np.load("resnetoutput_img_without_extreme_val.npy")
plt.subplot(2, 2, 1)
ax = sns.heatmap(resnet_img)
plt.title("ResNet-50 with extreme value")
plt.subplot(2, 2, 2)
ax = sns.heatmap(swin_img)
plt.title("Swin-T with extrame value")
plt.subplot(2, 2, 3)
ax = sns.heatmap(resnet_img_no_extreme)
plt.title("ResNet-50 w/o extreme value")
plt.subplot(2, 2, 4)
ax = sns.heatmap(swin_img_no_extreme)
plt.title("Swin-T w/o extrame value")
fig = plt.gcf()
fig.set_size_inches((11.97, 8.36), forward=False)
# fig.set_size_inches((12.31, 8.71), forward=False)
# fig.savefig("feture_map_subplots_13inches.pdf", dpi=300)
# fig.savefig("feture_map_subplots_13inches.eps", dpi=400)
fig.tight_layout()
plt.tight_layout()
plt.savefig("feture_map_subplots_13inches_200dpi.png", dpi=200)

plt.show()

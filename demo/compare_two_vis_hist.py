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

# Hist
conv_color = "#66bd63"
trans_color = "#f46d43"
swin_hist = np.load("swinlast_np_normalize_flat.npy")
resnet_hist = np.load("resnetlast_np_normalize_flat.npy")
swin_hist_no_extreme = np.load("swinlast_np_normalize_flat_no_extreme.npy")
resnet_hist_no_extreme = np.load("resnetlast_np_normalize_flat_no_extreme.npy")
"""
plt.hist(swin_hist, bins=np.arange(0, 1, 0.005), color =trans_color,density=False, label ="Swin-T" )
plt.hist(resnet_hist, bins=np.arange(0, 1, 0.005), color =conv_color, density=False, label="ResNet-50")
plt.title("Histogram of backbone last layer feature map")
plt.legend()
#plt.savefig("hist.png", dpi = 300)  
plt.show()
"""
ALPHA = 0.5
begin = 0
end = 0.7
plt.close()
plt.subplot(2, 1, 1)
plt.hist(
    swin_hist,
    bins=np.arange(begin, 1, 0.005),
    color=trans_color,
    density=False,
    label="Swin-T",
    alpha=ALPHA,
)
plt.hist(
    resnet_hist,
    bins=np.arange(begin, 1, 0.005),
    color=conv_color,
    density=False,
    label="ResNet-50",
    alpha=ALPHA,
)
plt.title("Feature map histogram with extreme value")
plt.legend()
plt.subplot(2, 1, 2)
plt.hist(
    swin_hist_no_extreme,
    bins=np.arange(begin + 0.001, 1, 0.002),
    color=trans_color,
    density=False,
    label="Swin-T",
    alpha=ALPHA,
)
plt.hist(
    resnet_hist_no_extreme,
    bins=np.arange(begin + 0.001, 1, 0.002),
    color=conv_color,
    density=False,
    label="ResNet-50",
    alpha=ALPHA,
)

"""
#plt.hist(swin_hist, bins=np.arange(begin+0.001, end, 0.001), color =trans_color,density=False, label ="Swin-T", alpha=ALPHA )
#plt.hist(resnet_hist, bins=np.arange(begin+0.001, end, 0.001), color =conv_color, density=False, label="ResNet-50", alpha=ALPHA)
"""
plt.title("Feature map histogram w/o extreme value")
plt.legend()
fig = plt.gcf()
fig.set_size_inches((11.97, 8.36), forward=False)
plt.tight_layout()
plt.savefig("hist_new_c.png", dpi=300)
plt.show()

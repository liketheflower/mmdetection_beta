import pandas as pd
import matplotlib.pyplot as plt
import os

"""
"/Users/jimmy/repos/jimmy/YONOD/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_RGB_load_from/20220211_002440.log.json  /Users/jimmy/repos/jimmy/InterTrans/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_load_from/20220203_003649.log.json /Users/jimmy/repos/jimmy/YONOD/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_sunrgbd_rgb_dhs_mixed/cc.log.json"

"""
rgb_dhs_mixed_on_rgb_res = [0.165, 0.188, 0.201, 0.202, 0.205, 0.206, 0.212, 0.211, 0.225, 0.213, 0.208, 0.218, 0.217, 0.226, 0.222, 0.222, 0.226, 0.214, 0.223, 0.215, 0.215, 0.223, 0.222, 0.215, 0.21, 0.213, 0.217, 0.237, 0.243, 0.246, 0.244, 0.246, 0.246, 0.245, 0.245, 0.245, 0.244, 0.244, 0.245, 0.244, 0.244, 0.245, 0.246, 0.244, 0.244, 0.243, 0.244, 0.245, 0.243, 0.244, 0.245, 0.244, 0.246, 0.246, 0.245, 0.243, 0.244, 0.243, 0.244, 0.244, 0.245, 0.245, 0.245, 0.244, 0.245, 0.246, 0.244, 0.244, 0.244, 0.246, 0.244, 0.246, 0.244, 0.244, 0.245, 0.246, 0.247, 0.246, 0.247, 0.245, 0.248, 0.245, 0.245, 0.245, 0.245, 0.246, 0.245, 0.247, 0.245, 0.245, 0.247, 0.245, 0.245, 0.244, 0.246, 0.245, 0.244, 0.246, 0.245, 0.247]

conv_color = "#66bd63"
trans_color = "#f46d43"
gray_color = "#878787"
conv_color_lighter = "#878787"
trans_color_lighter = "#abd9e9"
EPOCHS = 50
log_experiment_name_map = {
    "mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_RGB_load_from": "rgb only model",
    "mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_load_from": "dhs only model",
    "mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_sunrgbd_rgb_dhs_mixed": "unified model test on dhs",
}


def get_experiment_name(log_path):
    """
    Infer the experiment name from the log path.
    """
    key = log_path.split("/")[-2]
    return log_experiment_name_map[key]


keys = [
    "rgb only model",
    "dhs only model",
    "unified model test on rgb",
    "unified model test on dhs",
]

markers = {
    "rgb only model": "o",
    "dhs only model": "s",
    "unified model test on rgb": "o",
    "unified model test on dhs": "s",
}
markersize = 4
markersizes = {
    "rgb only model": markersize,
    "dhs only model": markersize,
    "unified model test on rgb": markersize,
    "unified model test on dhs": markersize,
}
colors = {
    "rgb only model": gray_color,
    "dhs only model": gray_color,
    "unified model test on rgb": trans_color,
    "unified model test on dhs": conv_color,
}
alphas = {
    "rgb only model": 0.99,
    "dhs only model": 0.99,
    "unified model test on rgb": 0.99,
    "unified model test on dhs": 0.99,
}
linea = "solid"
lineb = "dashed"
linestyles = {
    keys[0]: lineb,
    keys[1]: lineb,
    keys[2]: linea,
    keys[3]: linea,
}


def vis_mAP_bbox(res_fn):
    df = pd.read_pickle(res_fn)
    print(list(df["legend"]))
    df["experiment_name"] = df["legend"].apply(get_experiment_name)
    for i, (exp_name, xs, ys) in enumerate(zip(df["experiment_name"], df["x_values"], df["y_values"])):
        # plt.plot(xs, ys, marker=markers[exp_name], c = colors[exp_name], alpha=alphas[exp_name], label=exp_name)
        if i == 2:continue
        xs = xs[:EPOCHS]
        ys = ys[:EPOCHS]
        plt.plot(
            xs,
            ys,
            marker=markers[exp_name],
            markersize=markersizes[exp_name],
            linestyle=linestyles[exp_name],
            c=colors[exp_name],
            alpha=alphas[exp_name],
            label=exp_name,
        )
    rgb_dhs_mixed_on_rgb_xs = list(range(1, EPOCHS+1))
    rgb_dhs_mixed_on_rgb_ys = rgb_dhs_mixed_on_rgb_res[:EPOCHS]
    exp_name = keys[2]
    plt.plot(
	rgb_dhs_mixed_on_rgb_xs,
	rgb_dhs_mixed_on_rgb_ys,
	marker=markers[exp_name],
	markersize=markersizes[exp_name],
	linestyle=linestyles[exp_name],
	c=colors[exp_name],
	alpha=alphas[exp_name],
	label=exp_name,
    )
    rgb_dhs_mixed_on_rgb_xs = list(range(EPOCHS))
    rgb_dhs_mixed_on_rgb_ys = rgb_dhs_mixed_on_rgb_res[:EPOCHS]
    
    for i, (exp_name, xs, ys) in enumerate(zip(df["experiment_name"], df["x_values"], df["y_values"])):
        # plt.plot(xs, ys, marker=markers[exp_name], c = colors[exp_name], alpha=alphas[exp_name], label=exp_name)
        if i != 2:continue
        xs = xs[:EPOCHS]
        ys = ys[:EPOCHS]
        plt.plot(
            xs,
            ys,
            marker=markers[exp_name],
            markersize=markersizes[exp_name],
            linestyle=linestyles[exp_name],
            c=colors[exp_name],
            alpha=alphas[exp_name],
            label=exp_name,
        )
    
    plt.legend()
    plt.xlabel("epoch")
    # plt.ylabel("mAP@.5:.95 for SUN RGB-D 79 categories")
    plt.ylabel("mAP at IoU=.50")
    plt.title("SUN RGB-D 79 detection performance")
    plt.tight_layout()
    plt.savefig("mAP_bbox.png", dpi=400)
    plt.show()


if __name__ == "__main__":
    vis_mAP_bbox("./results/bbox_mAP_50.pkl")

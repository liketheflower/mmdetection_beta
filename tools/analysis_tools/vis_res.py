import pandas as pd
import matplotlib.pyplot as plt
import os

"""
['/Users/jimmy/repos/jimmy/simCrossTrans/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_load_from/cc.log.json_bbox_mAP', '/Users/jimmy/repos/jimmy/simCrossTrans/work_dirs/mask_rcnn_r50_fpn_mstrain-poly_3x_coco/cc.log.json_bbox_mAP', '/Users/jimmy/repos/jimmy/simCrossTrans/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_no_pretrain/cc.log.json_bbox_mAP', '/Users/jimmy/repos/jimmy/simCrossTrans/work_dirs/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_no_pretrain/cc.log.json_bbox_mAP']
"""


conv_color = "#66bd63"
trans_color = "#f46d43"
gray_color = "#878787"
conv_color_lighter = "#878787"
trans_color_lighter = "#abd9e9"
log_experiment_name_map = {
    "mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_load_from": "Swin-T with simCrossTrans",
    "mask_rcnn_r50_fpn_mstrain-poly_3x_coco": "ResNet-50 with simCrossTrans",
    "mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_no_pretrain": "Swin-T w/o simCrossTrans",
    "mask_rcnn_r50_fpn_mstrain-poly_3x_coco_no_pretrain": "ResNet-50 w/o simCrossTrans",
}


def get_experiment_name(log_path):
    """
    Infer the experiment name from the log path.
    """
    key = log_path.split("/")[-2]
    return log_experiment_name_map[key]


markers = {
    "Swin-T with simCrossTrans": "o",
    "Swin-T w/o simCrossTrans": "o",
    "ResNet-50 with simCrossTrans": "s",
    "ResNet-50 w/o simCrossTrans": "s",
}
markersize = 3
markersizes = {
    "Swin-T with simCrossTrans": markersize,
    "Swin-T w/o simCrossTrans": markersize,
    "ResNet-50 with simCrossTrans": markersize,
    "ResNet-50 w/o simCrossTrans": markersize,
}
colors = {
    "Swin-T with simCrossTrans": trans_color,
    "Swin-T w/o simCrossTrans": trans_color_lighter,
    "ResNet-50 with simCrossTrans": conv_color,
    "ResNet-50 w/o simCrossTrans": conv_color_lighter,
}
alphas = {
    "Swin-T with simCrossTrans": 0.8,
    "Swin-T w/o simCrossTrans": 0.55,
    "ResNet-50 with simCrossTrans": 0.8,
    "ResNet-50 w/o simCrossTrans": 0.55,
}
linea = "solid"
lineb = "dashed"
linestyles = {
    "Swin-T with simCrossTrans": linea,
    "Swin-T w/o simCrossTrans": lineb,
    "ResNet-50 with simCrossTrans": linea,
    "ResNet-50 w/o simCrossTrans": lineb,
}


def vis_mAP_bbox(res_fn):
    df = pd.read_pickle(res_fn)
    print(list(df["legend"]))
    df["experiment_name"] = df["legend"].apply(get_experiment_name)
    for exp_name, xs, ys in zip(df["experiment_name"], df["x_values"], df["y_values"]):
        # plt.plot(xs, ys, marker=markers[exp_name], c = colors[exp_name], alpha=alphas[exp_name], label=exp_name)
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
    plt.ylabel("mAP at IoU=.50:.05:.95")
    plt.title("SUN RGB-D 79 categories detection performance")
    plt.tight_layout()
    plt.savefig("mAP_bbox.png", dpi=400)
    plt.show()


if __name__ == "__main__":
    vis_mAP_bbox("./results/bbox_mAP.pkl")

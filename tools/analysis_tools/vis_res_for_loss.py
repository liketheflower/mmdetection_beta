import pandas as pd
import matplotlib.pyplot as plt
import os

"""
['/Users/jimmy/repos/jimmy/InterTrans/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_load_from/cc.log.json_bbox_mAP', '/Users/jimmy/repos/jimmy/InterTrans/work_dirs/mask_rcnn_r50_fpn_mstrain-poly_3x_coco/cc.log.json_bbox_mAP', '/Users/jimmy/repos/jimmy/InterTrans/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_no_pretrain/cc.log.json_bbox_mAP', '/Users/jimmy/repos/jimmy/InterTrans/work_dirs/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_no_pretrain/cc.log.json_bbox_mAP']
"""


conv_color = "#66bd63"
trans_color = "#f46d43"
gray_color = "#878787"
conv_color_lighter = "#878787"
trans_color_lighter = "#abd9e9"
log_experiment_name_map = {
    "mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_load_from": "Swin-T with InterTrans",
    "mask_rcnn_r50_fpn_mstrain-poly_3x_coco": "ResNet-50 with InterTrans",
    "mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_no_pretrain": "Swin-T w/o InterTrans",
    "mask_rcnn_r50_fpn_mstrain-poly_3x_coco_no_pretrain": "ResNet-50 w/o InterTrans",
}


def get_experiment_name(log_path):
    """
    Infer the experiment name from the log path.
    """
    key = log_path.split("/")[-2]
    return log_experiment_name_map[key]


markers = {
    "Swin-T with InterTrans": "o",
    "Swin-T w/o InterTrans": "o",
    "ResNet-50 with InterTrans": "s",
    "ResNet-50 w/o InterTrans": "s",
}
markersize = 4.2
markersizes = {
    "Swin-T with InterTrans": markersize,
    "Swin-T w/o InterTrans": markersize,
    "ResNet-50 with InterTrans": markersize,
    "ResNet-50 w/o InterTrans": markersize,
}
colors = {
    "Swin-T with InterTrans": trans_color,
    "Swin-T w/o InterTrans": trans_color_lighter,
    "ResNet-50 with InterTrans": conv_color,
    "ResNet-50 w/o InterTrans": conv_color_lighter,
}
alphas = {
    "Swin-T with InterTrans": 1,
    "Swin-T w/o InterTrans": 1,
    "ResNet-50 with InterTrans": 1,
    "ResNet-50 w/o InterTrans": 1,
}
linestyles = {
    "Swin-T with InterTrans": "solid",
    "Swin-T w/o InterTrans": "solid",
    "ResNet-50 with InterTrans": "solid",
    "ResNet-50 w/o InterTrans": "solid",
}
alpha = 0.85


def alpha_filter(ys):
    new_ys = [ys[0]]
    for i in range(1, len(ys)):
        new_ys.append(alpha * new_ys[-1] + ys[i] * (1 - alpha))
    return new_ys


def vis_loss(res_fn, loss_name=""):
    df = pd.read_pickle(res_fn)
    df["y_values_alpha_filter"] = df["y_values"].apply(alpha_filter)
    print(list(df["legend"]))
    df["experiment_name"] = df["legend"].apply(get_experiment_name)
    for exp_name, xs, ys in zip(
        df["experiment_name"], df["x_values"], df["y_values_alpha_filter"]
    ):
        # plt.plot(xs, ys, marker=markers[exp_name], c = colors[exp_name], alpha=alphas[exp_name], label=exp_name)
        plt.plot(
            xs,
            ys,
            linestyle=linestyles[exp_name],
            c=colors[exp_name],
            alpha=alphas[exp_name],
            label=exp_name,
        )
    plt.legend()
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title(" ".join(loss_name.split("_")))
    plt.savefig(loss_name + ".eps", dpi=500)
    plt.show()


def vis_loss_subplots(res_fn, loss_name="", sub_plot_id=0, remove_outliter=True):
    df = pd.read_pickle(res_fn)
    df["y_values_alpha_filter"] = df["y_values"].apply(alpha_filter)
    print(list(df["legend"]))
    df["experiment_name"] = df["legend"].apply(get_experiment_name)
    plt.subplot(2, 2, sub_plot_id)
    for exp_name, xs, ys in zip(
        df["experiment_name"], df["x_values"], df["y_values_alpha_filter"]
    ):
        # plt.plot(xs, ys, marker=markers[exp_name], c = colors[exp_name], alpha=alphas[exp_name], label=exp_name)
        N = 2 if remove_outliter else 0
        # change unit to K
        xs = [v / 1000 for v in xs]
        plt.plot(
            xs[N:],
            ys[N:],
            linestyle=linestyles[exp_name],
            c=colors[exp_name],
            alpha=alphas[exp_name],
            label=exp_name,
        )
    if sub_plot_id == 2:
        plt.legend()
    if sub_plot_id in [3, 4]:
        plt.xlabel("iteration (K)")
    if sub_plot_id in [1, 3]:
        plt.ylabel("loss")
    plt.title(" ".join(loss_name.split("_")))
    # plt.savefig(loss_name + ".eps", dpi=500)
    # plt.show()


# loss_rpn_cls": 0.09454, "loss_rpn_bbox": 0.04825, "loss_cls": 0.48573, "acc": 90.14453, "loss_bbox": 0.35008, "loss_mask": 0.0, "loss": 0.9786, "time": 0.33392}
if __name__ == "__main__":
    """
    vis_loss("./results/loss_rpn_cls.pkl", "RPN_classification_loss")
    vis_loss("./results/loss_rpn_bbox.pkl", "RPN_BBOX_loss")
    vis_loss("./results/loss_cls.pkl", "Classification_loss")
    vis_loss("./results/loss_bbox.pkl", "BBOX_loss")
    plt.close()
    """
    vis_loss_subplots("./results/loss_rpn_cls.pkl", "RPN_classification_loss", 1)
    vis_loss_subplots("./results/loss_rpn_bbox.pkl", "RPN_BBOX_loss", 2)
    vis_loss_subplots("./results/loss_cls.pkl", "Classification_loss", 3)
    vis_loss_subplots("./results/loss_bbox.pkl", "BBOX_loss", 4)
    plt.tight_layout()
    fig = plt.gcf()
    fig.set_size_inches((11.97, 8.36), forward=False)
    # fig.set_size_inches((12.31, 8.71), forward=False)
    fig.savefig("loss_subplots_13inches.eps", dpi=500)

    # plt.savefig("loss_subplots.eps", dpi=500)
    plt.show()

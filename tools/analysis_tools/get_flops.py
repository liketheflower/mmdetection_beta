# Copyright (c) OpenMMLab. All rights reserved.
import argparse

import numpy as np
import torch
from mmcv import Config, DictAction

from mmdet.models import build_detector
import torchvision.models as models
from ptflops import get_model_complexity_info

"""
try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError("Please upgrade mmcv to >0.6.2")
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    parser.add_argument("config", help="train config file path")
    #parser.add_argument(
    #    "--shape", type=int, nargs="+", default=[1280, 800], help="input image size"
    #)
    parser.add_argument(
        "--shape", type=int, nargs="+", default=[800, 1120], help="input image size"
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file. If the value to "
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        "Note that the quotation marks are necessary and that no white space "
        "is allowed.",
    )
    parser.add_argument(
        "--size-divisor",
        type=int,
        default=32,
        help="Pad the input image, the minimum size that is divisible "
        "by size_divisor, -1 means do not pad the image.",
    )
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    if len(args.shape) == 1:
        h = w = args.shape[0]
    elif len(args.shape) == 2:
        h, w = args.shape
    else:
        raise ValueError("invalid input shape")
    orig_shape = (3, h, w)
    divisor = args.size_divisor
    if divisor > 0:
        h = int(np.ceil(h / divisor)) * divisor
        w = int(np.ceil(w / divisor)) * divisor

    input_shape = (3, h, w)

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    model = build_detector(
        cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg"))
    """
    if torch.cuda.is_available():
        model.cuda()
    """
    model.cpu()
    model.eval()

    if hasattr(model, "forward_dummy"):
        model.forward = model.forward_dummy
    else:
        raise NotImplementedError(
            "FLOPs counter is currently not currently supported with {}".format(
                model.__class__.__name__
            )
        )
    with torch.no_grad():
        macs, params = get_model_complexity_info(model, (3, 800, 1120), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    with torch.no_grad():
        flops, params = get_model_complexity_info(model, input_shape)
    split_line = "=" * 30

    if divisor > 0 and input_shape != orig_shape:
        print(
            f"{split_line}\nUse size divisor set input shape "
            f"from {orig_shape} to {input_shape}\n"
        )
    print(
        f"{split_line}\nInput shape: {input_shape}\n"
        f"Flops: {flops}\nParams: {params}\n{split_line}"
    )
    print(
        "!!!Please be cautious if you use the results in papers. "
        "You may need to check if all ops are supported and verify that the "
        "flops computation is correct."
    )


if __name__ == "__main__":
    main()

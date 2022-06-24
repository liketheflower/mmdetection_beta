# Swin -S
cfg=/data/sophia/a/Xiaoke.Shen54/repos/SwinTransFusion/work_dirs/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_sunrgbd_rgbdhs_mixed/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_sunrgbd_rgbdhs_mixed.py
python get_flops.py $cfg  > swin_s_flops.log 

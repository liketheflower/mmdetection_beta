img=/data/sophia/a/Xiaoke.Shen54/DATASET/sunrgbd_DO_NOT_DELETE/val/dhs/006223_dhs.png
# swin
#cfg=/data/sophia/a/Xiaoke.Shen54/repos/RGB2PC/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_load_from.py               
#checkpoint=/data/sophia/a/Xiaoke.Shen54/repos/RGB2PC/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_load_from/epoch_100.pth
#python image_demo_izzy.py $img $cfg $checkpoint --device=cpu 
# resnet with simCrossTrans
cfg=/data/sophia/a/Xiaoke.Shen54/repos/RGB2PC/configs/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco.py                        
#checkpoint=/data/sophia/a/Xiaoke.Shen54/repos/RGB2PC/work_dirs/mask_rcnn_r50_fpn_mstrain-poly_3x_coco/epoch_100.pth
python get_flops.py $cfg  > resnet_flops.log 

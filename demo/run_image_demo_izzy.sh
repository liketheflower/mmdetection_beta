cfg=/data/sophia/a/Xiaoke.Shen54/repos/RGB2PC/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_load_from.py               
checkpoint=/data/sophia/a/Xiaoke.Shen54/repos/RGB2PC/checkpoints/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth
python image_demo_izzy.py izzy.png $cfg $checkpoint --device=cpu 

# swin
cfg=/Users/jimmy/repos/jimmy/InterTrans/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_load_from.py  
python get_flops.py $cfg  > swin_flops.log 




# resnet with simCrossTrans
#cfg=/Users/jimmy/repos/jimmy/InterTrans/configs/mask_rcnn/mask_rcnn_r50_fpn_mstrain-poly_3x_coco.py                        
#python get_flops.py $cfg  > resnet_flops.log 

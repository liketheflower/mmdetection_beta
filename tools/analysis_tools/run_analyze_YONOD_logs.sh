<<comment
json_logs="/Users/jimmy/repos/jimmy/YONOD/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_RGB_load_from/20220211_002440.log.json  /Users/jimmy/repos/jimmy/YONOD/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_sunrgbd_rgb2dhs/20220212_073746.log.json" 
python analyze_logs.py plot_curve ${json_logs} --keys="bbox_mAP_50"
comment

json_logs="/Users/jimmy/repos/jimmy/YONOD/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_RGB_load_from/cc.log.json  /Users/jimmy/repos/jimmy/YONOD/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_sunrgbd_rgb2dhs/cc.log.json" 
python analyze_logs.py plot_curve ${json_logs} --keys="bbox_mAP_50"


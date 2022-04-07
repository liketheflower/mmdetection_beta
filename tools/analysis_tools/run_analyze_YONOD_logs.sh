json_logs="/Users/jimmy/repos/jimmy/YONOD/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_RGB_load_from/20220211_002440.log.json  /Users/jimmy/repos/jimmy/YONOD/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_sunrgbd_rgb2dhs/20220212_073746.log.json" 
#python analyze_logs.py cal_train_time ${json_logs}
#python analyze_logs.py plot_curve ${json_logs} 
#"loss_rpn_cls": 0.09454, "loss_rpn_bbox": 0.04825, "loss_cls": 0.48573, "acc": 90.14453, "loss_bbox": 0.35008, "loss_mask": 0.0, "loss": 0.9786, "time": 0.33392}
#python analyze_logs.py plot_curve ${json_logs} --keys="loss_rpn_cls"
#python analyze_logs.py plot_curve ${json_logs} --keys="loss_rpn_bbox"
#python analyze_logs.py plot_curve ${json_logs} --keys="loss_cls"
#python analyze_logs.py plot_curve ${json_logs} --keys="loss_bbox"
python analyze_logs.py plot_curve ${json_logs} --keys="bbox_mAP_50"


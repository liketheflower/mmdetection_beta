json_logs="/Users/jimmy/repos/jimmy/InterTrans/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_load_from/cc.log.json /Users/jimmy/repos/jimmy/InterTrans/work_dirs/mask_rcnn_r50_fpn_mstrain-poly_3x_coco/cc.log.json /Users/jimmy/repos/jimmy/InterTrans/work_dirs/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_no_pretrain/cc.log.json   /Users/jimmy/repos/jimmy/InterTrans/work_dirs/mask_rcnn_r50_fpn_mstrain-poly_3x_coco_no_pretrain/cc.log.json"
#python analyze_logs.py cal_train_time ${json_logs}
python analyze_logs.py plot_curve ${json_logs} --keys="loss_rpn_cls"


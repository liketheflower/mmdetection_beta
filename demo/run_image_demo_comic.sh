cfg=/data/sophia/a/Xiaoke.Shen54/repos/RGB2PC/configs/swin/mask_rcnn_swin-t-p4-w7_fpn_ms-crop-3x_coco_load_from.py               
checkpoint=/data/sophia/a/Xiaoke.Shen54/repos/RobustDetRL_YONOD/checkpoints/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco_20210908_165006-90a4008c.pth
img_dir=/data/sophia/a/Xiaoke.Shen54/repos/RobustDetRL/application/object_detection/test/comics/
log_path=./logs/
mkdir -p $log_path
ret_path="./comics_ret/"
mkdir -p $ret_path
for img in "$img_dir"*.jpg
do
  echo "$img"
  fn="$(basename $img)"
  echo $fn
  python image_demo_izzy.py $img $cfg $checkpoint --device=cuda > ${log_path}${fn}.log 
   mv result.jpg $fn
   mv $fn $ret_path 
done

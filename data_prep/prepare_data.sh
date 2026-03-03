#! /bin/bash

python3 /home/hxy/ultralytics/data_prep/annotation.py --dataset_dir /home/hxy/ultralytics/data/yolo --dataset_name knob_big --objs_num 9
python3 /home/hxy/ultralytics/data_prep/data_split.py --dataset_dir /home/hxy/ultralytics/data/yolo --dataset_name knob_big --train_ratio 0.9
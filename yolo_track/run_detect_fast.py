import os

l = os.listdir("./test/images")

for i in l:
    os.system(f"python detectfastreid.py --weights best_0530.pt --source /home/yixiang/anaconda3/envs/yolov9/yolov9/test/images/{i}/ --device '0' --name {i} --fuse-score --agnostic-nms --with-reid --save-txt")

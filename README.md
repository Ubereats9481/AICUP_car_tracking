# AICUP_car_tracking
AICUP 2024 spring AI驅動出行未來：跨相機多目標車輛追蹤競賽 (模型組)

## 重點程式說明

fastreid_train/fast_reid/fastreid/tools/train_net.py訓練FastReID模型

相關訓練參數檔案位於fastreid_train/logs/AICUP_115/bagtricks_R50-ibn_circleG25M35/config.yaml

yolo_track/train_dual.py訓練YOLOv9模型

訓練好後使用yolo_track/detectfastreid.py對測試資料進行預測，並且會將輸出自動轉為最終提交的格式

可使用yolo_track/run_detect_fast.py一次對所有測試資料進行預測，並且會將輸出自動轉為最終提交的格式

所需套件寫在fastreid_train/requirements.txt與yolo_track/requirements.txt檔案中
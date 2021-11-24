import os
os.getcwd()
from yolov4_function import read_class_names, draw_bbox, count_objects
from tensorflow.python.saved_model import tag_constants
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2

MODEL_PATH = './checkpoints/yolov4-416'
CLASS_NAME = "./data/classes/coco.names" # class 이름
DATA_NAME = 'data/cctv_csv/cctv_url_all.csv'
IOU_THRESHOLD = 0.45
SCORE_THRESHOLD = 0.25
INPUT_SIZE = 416

# load model
saved_model_loaded = tf.saved_model.load(MODEL_PATH, tags=[tag_constants.SERVING])
infer = saved_model_loaded.signatures['serving_default']

cctv_data = pd.read_csv(DATA_NAME)
cctv_data = cctv_data.drop('Unnamed: 0',axis=1)


for cctv, cctv_name in zip(cctv_data['0'], cctv_data['3']):
    box = [] # count 수를 받을 빈 box생성
    df_box = pd.DataFrame()
    
    ## 비디오 한프레임 한프레임를 캡처한다.
    cap = cv2.VideoCapture('%s'%str(cctv))
    # 한프레임 씩 캡처한 이미지를 반복문으로 루프 시킨다.
    while cap.isOpened(): #isOpened 카메라 프레임 읽기
        
        
        ret, img = cap.read() # isOpened한 값을 ret과 img에 담는다
        if not ret:
            break
        
        # 카메라의 색깔을 반전시킨다 RGB를 BGR로
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        
        
        # resize(winname(창이름), width(가로크기), height(세로크기))
        img_input = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        img_input = img_input / 255.
        img_input = img_input[np.newaxis, ...].astype(np.float32)
        
        # constamt img_input를 상수로 만들어 준다.
        img_input = tf.constant(img_input)
        pred_bbox = infer(img_input)
        for key, value in pred_bbox.items():
            boxes = value[:, :, 0:4]
            pred_conf = value[:, :, 4:]
            
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=IOU_THRESHOLD,
            score_threshold=SCORE_THRESHOLD)

        pred_bbox = [boxes.numpy(), scores.numpy(), classes.numpy(), valid_detections.numpy()]
        
        ######## count object ####################
        class_names = read_class_names(CLASS_NAME)
        allowed_classes = list(class_names.values())
        counted_classes = count_objects(pred_bbox, by_class = True, allowed_classes=allowed_classes)
        
        box.append([counted_classes])
        

        for key, value in counted_classes.items():
                print("Number of {}s: {}".format(key, value))
        
        result = draw_bbox(img, pred_bbox, 
                            # info=True, 
                            show_label=True,
                            counted_classes=counted_classes, 
                            allowed_classes=allowed_classes, 
                            read_plate=False)

        
        ## 처음에 색깔 반전 시킨걸 다시 원래대로 돌려준다.
        result = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)

        cv2.imshow('result', result)
        
        # 종료버튼
        if cv2.waitKey(1) == ord('q'):
            break
        
    df_box = pd.DataFrame(box)
    df_box.to_csv('data/countData/count%s.csv'%cctv_name)

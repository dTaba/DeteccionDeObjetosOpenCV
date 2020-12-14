import cv2 as cv

webcam = cv.VideoCapture(0)
nombresObjetos = []
archivoNombres = 'coco.names'
threshold = 0.5

with open(archivoNombres, 'rt') as f:
    nombresObjetos = f.read().rstrip('\n').split('\n')

config = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt' 
weight = 'frozen_inference_graph.pb'

net = cv.dnn_DetectionModel(weight, config)
net.setInputSize(320, 320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)



while True:
    ok, img = webcam.read()
    ids, confs, boxs = net.detect(img, threshold)
    
    if len(ids) > 0:
        for idObjeto, confidence, box in zip(ids.flatten(), confs.flatten(), boxs):
            cv.rectangle(img, box, (0,0,255), thickness=2)
            cv.putText(img, nombresObjetos[idObjeto - 1],(box[0] + 50, box[1] + 30), cv.FONT_HERSHEY_COMPLEX, 1.00, (0,0,255), thickness=2)
            cv.putText(img, str(round(confidence*100)) + "%",(box[0] + 50, box[1] + 80), cv.FONT_HERSHEY_COMPLEX, 1.00, (0,0,255), thickness=2)
    
    cv.imshow("Webcam", img)
    
    cv.waitKey(1)
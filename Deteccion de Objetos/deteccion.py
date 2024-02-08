#pip install opencv-python
import cv2
import numpy as np 

model = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = model.getUnconnectedOutLayersNames()

img= cv2.imread("descarga.jpeg")
height, width = img.shape[:2]

blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False )
model.setInput(blob)
outputs=model.forward(layer_names)

for output in outputs:
    for detection in output:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)
            x = int(center_x - w/2)
            y = int(center_y - h/2)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("Objeto Detectado", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


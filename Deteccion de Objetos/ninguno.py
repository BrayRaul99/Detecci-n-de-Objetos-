import cv2
import numpy as np 

model = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layers_names = model.getUnconnectedOutLayersNames()

with open("coco.names", "r") as f:
        labels = f.read().strip().split("\n")

img = cv2.imread("descarga.jpeg")
height, width = img.shape[:2]

blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
model.setInput(blob)
outputs = model.forward(layers_names)

boxes = []
confidences = []
class_ids = []

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
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

if len(indices) > 0:
        for i in indices.flatten():
                x, y, w, h = boxes[i]
                label = str(labels[class_ids[i]])
                confidence = confidences[i]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, f"{label}: {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("Deteccion de objetos", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

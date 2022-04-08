###################################
# YOLO 이미지 처리
import cv2
import numpy as np

# 최소 신뢰도
min_confidence = 0.5

# Load Yolo
net = cv2.dnn.readNet("backup/yolov3.weights", "backup/yolov3.cfg")
# yolov3cfg = '/Users/jongukyang/Test_OpenCV_DeepLearning/Test_YOLO_Learn/dmgcar/custom-train-yolo.cfg'
# yolov3wig = '/Users/jongukyang/Test_OpenCV_DeepLearning/Test_YOLO_Learn/dmgcar/yolov33.weights'
# net = cv2.dnn.readNet(yolov3wig, yolov3cfg)

classes = []
# with open("coco.names", "r") as f:
with open("backup/classes.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
# sampleimg = '/Users/jongukyang/Test_OpenCV_DeepLearning/Test_YOLO_Learn/dmgcar/0121.JPEG'
img = cv2.imread("1.jpeg")
# img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape
cv2.imshow("Original Image", img)

# Detecting objects
blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False) # 320 x 320 / 416 x 416 / 609 x 609

net.setInput(blob)
outs = net.forward(output_layers)

# Showing informations on the screen
class_ids = [] # 클래스 아이디
confidences = [] # 정확도
boxes = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > min_confidence:
            # Object detected
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[3] * height)

            # Rectangle coordinates
            x = int(center_x - w / 2) # 사각형의 왼쪽 꼭짓점
            y = int(center_y - h / 2)

            boxes.append([x, y, w, h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

# NMSBoxes = 혼잡도 제거 / 노이즈 제거 / 얼굴을 인식하는 여러개의 박스를 하나로 만드는거임
indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)

font = cv2.FONT_HERSHEY_PLAIN
for i in range(len(boxes)):
    if i in indexes:
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        print(i, label) # 콘솔창에 출력
        color = colors[1]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, label, (x, y + 30), font, 1, (0, 255, 0), 1)


cv2.imshow("YOLO Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

################################33###
#
# # YOLO 동영상 처리
# import cv2
# import numpy as np
# import time
#
# file_name = 'yolo_01.mp4'
# min_confidence = 0.5
#
# def detectAndDisplay(frame):
#     start_time = time.time()
#     img = cv2.resize(frame, None, fx=0.4, fy=0.4)
#     height, width, channels = img.shape
#     cv2.imshow("Original Image", img)
#
#     # Detecting objects
#     blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
#
#     net.setInput(blob)
#     outs = net.forward(output_layers)
#
#     # Showing informations on the screen
#     class_ids = []
#     confidences = []
#     boxes = []
#
#     for out in outs:
#         for detection in out:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > min_confidence:
#                 # Object detected
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#
#                 # Rectangle coordinates
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)
#
#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
#     font = cv2.FONT_HERSHEY_PLAIN
#     for i in range(len(boxes)):
#         if i in indexes:
#             x, y, w, h = boxes[i]
#             label = "{}: {:.2f}%".format(classes[class_ids[i]], confidences[i] * 100)
#             print(i, label)
#             color = colors[i]
#             cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
#             cv2.putText(img, label, (x, y - 5), font, 1, color, 1)
#     end_time = time.time()
#     process_time = end_time - start_time
#     print("=== A frame took {:.3f} seconds".format(process_time))
#     cv2.imshow("YOLO Video", img)
#
#
# # Load Yolo
# net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
# classes = []
# with open("coco.names", "r") as f:
#     classes = [line.strip() for line in f.readlines()]
# layer_names = net.getLayerNames()
# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
# colors = np.random.uniform(0, 255, size=(len(classes), 3))
#
# # -- 2. Read the video stream
# cap = cv2.VideoCapture(file_name)
# if not cap.isOpened:
#     print('--(!)Error opening video capture')
#     exit(0)
# while True:
#     ret, frame = cap.read()
#     if frame is None:
#         print('--(!) No captured frame -- Break!')
#         break
#     detectAndDisplay(frame) # 프레임 단위로 전송
#     # Hit 'q' on the keyboard to quit!
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cv2.destroyAllWindows()

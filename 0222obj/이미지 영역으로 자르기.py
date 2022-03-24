import cv2
import numpy as np

# Yolo 로드
"""
# OpenCV로 딥러닝을 실행하기 위해서는 우선 cv2.dnn_Net 클래스 객체를 생성해야 한다.
# 객체 생성에는 훈련된 가중치와 네트워크 구성을 저장하고 있는 파일이 필요
# cv2.dnn.readNet(model, config=None, framework=None) -> retval(return value)
# model: 훈련된 가중치를 저장하고 있는 이진 파일 이름
# config: 네트워크 구성을 저장하고 있는 텍스트 파일 이름, config가 없는 경우도 많습니다.
# framework: 명시적인 딥러닝 프레임워크 이름 ex) framework = "tensorflow"
# retval: cv2.dnn_Net 클래스 객체
"""
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg", framework=None)
# net = cv2.dnn.readNet("custom-train-yolotiny_10000.weights", "custom-train-yolovtiny.cfg", framework=None)
classes = [] #리스트
# with : 파일을 열고 끝나면 닫는거
# coco.names
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
# print("net.getUnconnectedOutLayer():",net.getUnconnectedOutLayers())
layer_names = net.getLayerNames() #net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))
# print("layer_names:",layer_names)
print("output_layers:",output_layers) #output_layers: ['yolo_82', 'yolo_94', 'yolo_106']

# 이미지 가져오기
# img = cv2.imread("jongukposter.jpeg", cv2.IMREAD_COLOR) # 이미지 컬러로 출력
img = cv2.imread("sample2.jpg", cv2.IMREAD_COLOR) # 이미지 컬러로 출력
# img = cv2.VideoCapture(0) # 카메라 영상처리
# print(img.shape) # 이미지는 3차원 행렬로 return / sample.jpg는 3024 X 3024 사이즈
img = cv2.resize(img, dsize=None, fx=0.4, fy=0.4, interpolation=cv2.INTER_CUBIC) #사이즈 재조정 (입력이미지/절대크기/상대크기/보간법)
height, width, channels = img.shape #3차원
print('height :', height, 'width :', width, 'channels :', channels)

# Detecting objects
"""
# 네트워크 입력 블롭(blob) 만들기 - cv2.dnn.blobFromImage
입력 영상을 블롭(blob)객체로 만들어서 추론을 진행해야 합니다.
주의할 점은 인자들을 입력할 때 모델 파일이 어떻게 학습되었는지 파악하고 그에 맞게 입력을 해줘야함
하나의 영상을 추론할 때는 cv2.dnn.blobFromImage 함수를 이용하여 1개의 블롭 객체를 받고
여러 개의 영상을 추론할 때는 cv2.dnn.blobFromImages 함수로 여러 개의 블롭 객체를 받아서 샤용

# cv2.dnn.blobFromImage(image, scalefactor=None, size=None, mean=None, swapRB=None, crop=None, ddepth=None) -> retval
# image : 입력 이미지
# scalefactor : 입력 영상 픽셀 값에 곱할 값. 기본값은 1.
  - scalefactor은 딥러닝 학습을 진행할 때 입력 영상을 0~255 픽셀값을 이용했는지, 
    0~1로 정규화해서 이용했는지에 맞게 지정해줘야 합니다. 0~1로 정규화하여 학습을 진행했으면 1/255를 입력해줘야 합니다.
# size : 출력 영상의 크기. 기본값은 (0,0)
  - 학습할 때 사용한 영상의 크기를 입력. 그 size로 resize를 해주어 출력해야 함
# mean : 입력 영상 각 채널에서 뺄 평균 값. 기본값은 (0,0,0,0)
  - 학습할 때 mean 값을 빼서 계산한 경우 그와 동일한 mean 값을 지정
# swapRB : R과 B 채널을 서로 바꿀 것인지를 결정하는 플래그. 기본값은 False
  - BGR에서 R값과 B값을 바꿀 것인지를 결정
  - ** OpenCV에선 색을 표현할 경우 BGR 순으로 표현합니다! **
# crop : 크롭(crop)수행 여부. 기본값은 False
  - 학습할 때 영상을 잘라서 학습하였으면 그와 동일하게 입력해야 함 
# ddepth : 출력 블롭의 깊이. CV_32F 또는 CV_8U. 기본값은 CV_32F
  - 대부분의 경우 CV_32F를 사용
# retbal : 영상으로 부터 구한 블롭 객체. numpy.ndarray.shape=(N,C,H,W).dtype=numpy.float32.
  - 반환값의 shape=(N,C,H,W)인데 N은 갯수, C는 채널 갯수, H,W는 영상 크기를 의미
"""
blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(416, 416), mean=(0, 0, 0, 0), swapRB=True, crop=False)
print('type : ', type(blob))
print('shape :', blob.shape)
print('size :', blob.size)
"""
# 네트워크 입력 설정하기 - cv2.dnn_Net.setInput
# cv2.dnn_Net.setInput(blob, name=None, scalefactor=None, mean=None) -> None
# blob : 블롭 객체
# name : 입력 레이어 이름
# scalefactor : 추가적으로 픽셀 값에 곱할 값
# mean : 추가적으로 픽셀 값에서 뺄 평균 값
말 그대로 신경망에 넣을 사진만 setting 해준다.
"""
net.setInput(blob)
"""
# 네트워크 순방향 실행(추론) - cv2.dnn_Net.forward
추론을 진행할 때 사용하는 함수. 네트워크를 어떻게 생성했냐에 따라 출력을 여러 개 지정할 수 있음(outputNames)
# cv2.dnn_Net.forward(outputName=None) -> retval
# cv2.dnn_Net.forward(outputNames=None, outputBlobs=None) -> outputBlobs
# outputName : 출력 레이어 이름
# retval : 지정한 레이어의 블롭. 네트워크마다 다르게 결정됨.
# outputNames : 출력 레이어 이름 리스트
# outputBlobs : 지정한 레이어의 출력 블롭 리스트
--> Object Detection 수행하여 결과를 outs으로 반환
"""
outs = net.forward(output_layers) # interface를 돌려서 원하는 layer의 Feature Map 정보만 뽑아냄
print("outs type : list // outs length :", len(outs))
print("outs[0] : 첫번째 FeatureMap 13x13x85, outs[1] : 두번째 FeatureMap 26x26x85")

# 정보를 화면에 표시
# 원본 이미지를 네트웍에 입력시에는 (416, 416)로 resize 함.
# 이후 결과가 출력되면 resize된 이미지 기반으로 bounding box 위치가 예측 되므로 이를 다시 원복하기 위해 원본 이미지 shape정보 필요
rows = img.shape[0] # = width
cols = img.shape[1] # = height
print('rows :',rows, 'cols :', cols)

conf_threshold = 0.5  # confidence_
nms_threshold = 0.4   # 이 값이 클 수록 box가 많이 사라짐. 조금만 겹쳐도 NMS로 둘 중 하나 삭제하므로

class_ids = []
confidences = []
boxes = []

# 3개의 개별 output layer별로 Detect된 Object들에 대해서 Detection 정보 추출 및 시각화
for ix, output in enumerate(outs):
    print('output shape:', output.shape)
    # Detected된 Object별 iteration
    for jx, detection in enumerate(output):
        # class score는 detetection배열에서 5번째 이후 위치에 있는 값. 즉 6번쨰~85번째 까지의 값
        scores = detection[5:]
        # scores배열에서 가장 높은 값을 가지는 값이 class confidence, 그리고 그때의 위치 인덱스가 class id
        class_id = np.argmax(scores)
        confidence = scores[class_id] # 5번쨰 값은 objectness score이다. 객체인지 아닌지의 확률이다. 6번쨰~85번째 까지의 값이 그 객체일 확률 값이다.

        # confidence가 지정된 conf_threshold보다 작은 값은 제외
        if confidence > conf_threshold:
            print('ix:', ix, 'jx:', jx, 'class_id', class_id, 'confidence:', confidence)
            # detection은 scale된 좌상단, 우하단 좌표를 반환하는 것이 아니라, detection object의 중심좌표와 너비/높이를 반환
            # 원본 이미지에 맞게 scale 적용 및 좌상단, 우하단 좌표 계산
            center_x = int(detection[0] * cols)
            center_y = int(detection[1] * rows)
            width = int(detection[2] * cols)
            height = int(detection[3] * rows)
            left = int(center_x - width / 2) # 좌측 x좌표
            top = int(center_y - height / 2) # 위쪽 y좌표
            # 3개의 개별 output layer별로 Detect된 Object들에 대한 class id, confidence, 좌표정보를 모두 수집
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([left, top, width, height])



#노이즈 제거
"""
# cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
# Box : 감지된 개체를 둘러싼 사각형의 좌표
# Label : 감지된 물체의 이름
# Confidence : 0에서 1까지의 탐지에 대한 신뢰도
conf_threshold = 0.5
nms_threshold = 0.4
"""
indexes = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
print("indexes",indexes)

# 사각형의 좌표값 알고 싶으면 이거 주석 해제
print("class name:", class_id)
print("Center x좌표:", center_x, "y좌표:", center_y)
print("박스 x좌표:", left, "y좌표:", top)

#화면에 표시하기
for i in range(len(boxes)):
    if i in indexes:
        print("boxes len", len(boxes))
        print("boxes", boxes)
        print("indexes", indexes)
        x, y, w, h = boxes[i]
        print("boxex[i], x y w h", boxes[i], i)
        object_index = classes.index(classes[class_ids[i]])
        if (object_index == 0):  # z == 56 은 chair(의자)를 의미함, 따라서 다른걸 찾고싶다면 56을 바꿔주면 되는데 # 찾고싶은 물체를 z로 칭한다면, 물체의 인덱스는 coco.names 파일에 있는 물체 이름 줄 번호 -1 을 하면됨
            label = "{} : {:0.4f}".format(str(classes[class_ids[i]]), confidences[i])
            color = colors[i]
            roi = img[y:y+h,x:x+w]
            cv2.imwrite("t1.jpg", roi)
            org_img = cv2.imread('t1.jpg')
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
            print("@@boxex[i], x y w h", boxes[i], i)
            print("@@indexes", indexes)


cv2.imshow("Image", img)
cv2.imwrite("test1.jpg", img)
cv2.imshow("t1.jpg", org_img)
cv2.waitKey(0) # 키 입력을 기다리는 함수
cv2.destroyAllWindows() # 키 입력되면 창 종료


import pymysql
import pandas as pd
import cv2
import numpy as np
import time

def object_detection(url, cursor, testpython, name_list, rent_id):
    file_name = url
    weights_file = "yolov3.weights"
    cfg_file = "yolov3.cfg"
    classes_names_file = "coco.names"

    # @ 시작 시간 / 걸린 시간 체크하기
    start_time = time.time()

    # ! Yolo Load
    net = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)  # cfg 파일이 먼저

    # ! 클래스(coco.names) 파일 열기
    with open(classes_names_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # @ 색과 폰트 color, font
    color_lists = np.random.uniform(0, 255, size=(len(classes), 3))  # 색 80가지 랜덤으로
    font = cv2.FONT_HERSHEY_PLAIN

    # ! object detect 검출 대상 이미지 불러오기
    img = cv2.imread(file_name, cv2.IMREAD_COLOR)  # 이미지를 컬러로 출력

    height = img.shape[0]
    width = img.shape[1]
    # cv2.imshow("original Img", img)  # 원본 이미지

    # ! Detecting Objects 객체 검출 시작
    # @ 네트워크 입력 블롭(blob) 만들기 - cv2.dnn.blobFromImage
    blob = cv2.dnn.blobFromImage(img, scalefactor=1 / 255.0, size=(416, 416), mean=(0, 0, 0, 0), swapRB=True,
                                 crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)  # interface를 돌려서 원하는 layer의 Feature Map 정보만 뽑아냄

    # @ 신뢰도 설정 (신뢰도가 50% 가 넘는 것들 선택한다 라는 뜻)
    min_confidence = 0.5  # 최소 신뢰도
    nms_threshold = 0.4  # nms 최소 신뢰도, 값이 클 수록 box가 많이 사라짐. 조금만 겹쳐도 NMS로 둘 중 하나 삭제하므로

    # ! 3개의 개별 output layer별로 Detect된 Object들에 대해서 Detection 정보 추출 및 시각화
    class_ids = []
    confidences = []
    boxes = []
    colors = []

    for ix, output in enumerate(outs):
        # Detected된 Object별 iteration(반복)
        for jx, detection in enumerate(output):
            # class score는 detetection배열에서 5번째 이후 위치에 있는 값. 즉 6번쨰~85번째 까지의 값
            scores = detection[5:]
            # scores배열에서 가장 높은 값을 가지는 값이 class confidence, 그리고 그때의 위치 인덱스가 class id
            class_id = np.argmax(scores)
            confidence = scores[class_id]  # 5번쨰 값은 objectness score이다. 객체인지 아닌지의 확률이다. 6번쨰~85번째 까지의 값이 그 객체일 확률 값이다.

            # confidence가 지정된 min_confidence 보다 작은 값을 제외
            if confidence > min_confidence:
                # detection은 scale된 좌상단, 우하단 좌표를 반환하는 것이 아니라, detection object의 중심좌표와 너비/높이를 반환
                # 원본 이미지에 맞게 scale 적용 및 좌상단, 우하단 좌표 계산(원본 이미지의 height, width 곱해줌)
                # @ Object Detected
                # print("detection 보기 ", detection[0:4], ix, jx)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                right_x = int(detection[2] * width)
                bottom_y = int(detection[3] * height)
                # @ Rectangle 좌표 계산
                left_x = int(center_x - right_x / 2)  # 좌측 x 좌표
                top_y = int(center_y - bottom_y / 2)  # 우측 y 좌표
                # 3개의 개별 output layer별로 Detect된 Object들에 대한 class id, confidence, 좌표정보를 모두 수집
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([left_x, top_y, right_x, bottom_y])
                colors.append(color_lists[class_id])
                # print("left_x :", left_x, "top_y :", top_y, "right_x :", right_x, "bottom_y :", bottom_y)

    # ! 노이즈 제거 (중복 박스 제거)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, nms_threshold)

    # ! 화면에 표시하기 위한 코드
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]  # 좌측 x, 상단 y, 우측 x, 하단 y 좌표
            object_lable = "{} : {:0.4f}".format(classes[class_ids[i]], confidences[i])
            color = colors[i]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, object_lable, (x, y), font, 3, color, 3)

    # @ 걸린 시간 계산
    end_time = time.time()
    process_time = end_time - start_time
    print("=== A frame took {:.3f} seconds".format(process_time))

    # ! 이미지 출력 저장
    cv2.imshow("Detected Img", img)

    save_dir = "C:\\Users\\User\\Desktop\\ObjectDetect-main\\Capstone-Obj\\dimg\\" + name_list
    cv2.imwrite(save_dir, img)

    # cv2.waitKey(0)  # 키 입력을 기다리는 함수
    # cv2.destroyAllWindows()  # 키 입력되면 창 닫기 종료
    # update_data()


def load_data():
    testpython = pymysql.connect(
        host='localhost',
        port=3306,
        user='root',
        password='1234',
        db='car',
        charset='utf8',
        local_infile=1
    )
    print("Database Connect")
    cursor = testpython.cursor(pymysql.cursors.DictCursor)

    sql = "select * from carimg"
    cursor.execute(sql)
    results = cursor.fetchall()
    print("Main results :", results) # 딕셔너리

    # 딥러닝 시작
    # rent id 의 폴더에 접근하기
    # detect 된 파일은 detect 이름을 붙여서 저장하고 데베에 덮어씌우기
    rent_id = 16 # 이거를 가져와야함 근데 어떻게 전달받는지 모르겠음
    name_list = ["test1.jpg", "test2.jpg", "test3.jpg", "test4.jpg", "test5.jpg", "test6.jpg", ]
    count = 0
    for result in results:
        print("result :",result)
        if rent_id == result['id']:
            # 딥러닝
            # 사진 url 로 접근하기
            print("i am 10")
            for key, values in result.items():
                # print("data :", key, values)
                if values != str(values): # 문자열인 경우에만 가져와야함 그걸 이렇게 쓰는게 맞나
                    continue
                elif values == str(values):
                    print("values :", values)
                    url = values
                    print("start obj detect !!!")
                    object_detection(url, cursor, testpython, name_list[count], rent_id)
                    count = count+1
                else:
                    break

    results = pd.DataFrame(results)
    print(results)
    print("load_data success")
    testpython.close()  # 연결 닫기
    print("Database Connect End")


def update_data():
    testpython = pymysql.connect(
        host='localhost',
        port=3306,
        user='root',
        password='1234',
        db='car',
        charset='utf8',
        local_infile=1 ) # DB 연동
    cursor = testpython.cursor() # 디폴트 커서 생성
    # 전달받은 고객의 rent_id 확인 후 저장
    # "UPDATE carimg SET id='/Users/jongukyang/Desktop/'"+  +"WHERE id = 16;"
    # sql = "INSERT INTO carimg(id, imgurl, imgurl2, imgurl3) VALUES(16, '/Users/jongukyang/Desktop/hello.jpeg', '/Users/jongukyang/Desktop/hello2.jpeg', );"
    rent_id ="16"
    sql = "UPDATE carimg SET imgurl="'/Users/jongukyang/Desktop/hello.jpeg'", imgurl2="'/Users/jongukyang/Desktop/hello2.jpeg'", imgrul3="'/Users/jongukyang/Desktop/hello3.jpeg'" WHERE id=" + (rent_id)
    cursor.execute(sql)
    testpython.commit()
    # print('rowcount: ', testpython.rowcount)
    testpython.close()  # 연결 닫기
    print("save_data success")


def load_id_data(rent_id):
    testpython = pymysql.connect(
        host='localhost',
        port=3306,
        user='root',
        password='1234',
        db='car',
        charset='utf8',
        local_infile=1
    )
    print("Database Connect")
    print("Load_id_data")
    cursor = testpython.cursor(pymysql.cursors.DictCursor)

    # 전달받은 고객의 rent_id 확인 후 url 불러오기
    sql = "select * from carimg where id=" + str(rent_id) + ";"
    cursor.execute(sql)
    result = cursor.fetchall()
    print("result :",result)
    df_result = pd.DataFrame(result)
    print(df_result)
    print("load_id_data success")

    # 각 행에 위치한 데이터 하나씩 뽑아내기
    sql2 = "select imgurl from carimg where id=" + str(rent_id) + ";"
    cursor.execute(sql2)
    result2 = cursor.fetchall()
    print("result2 :", result2)
    df_result2 = pd.DataFrame(result2)
    print(df_result2)
    print("load Single Data")

    testpython.close()  # 연결 닫기
    print("Database Connect End")





# 현재 DB에 저장된 사진을 rent_id 키로 가져옴
# 가져와서 딥러닝 수행 후 이미지 이름을 변경
# DB에 변경된 파일 저장
# 덮어쓰기 형식?
load_data() # 데이터 로드
# load_id_data(16)

# 변수 설정
# rent_id = 9

# load_id_data(rent_id)

# save_data()


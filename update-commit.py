import pymysql
import cv2
import numpy as np
import time

# obj_detect_dir_path = "C:/Users/nyh71/Desktop/ObjectDetect-main/userId/detectImg/"
obj_detect_dir_path = "C:/Users/nyh71/Desktop/ObjectDetect-main/rent/"

name_lists = ["", "beforeFront.png", "beforeBack.png", "beforeDriveFront.png", "beforeDriveBack.png",
              "beforePassengerFront.png", "beforePassengerBack.png", "afterFront.png",
              "afterBack.png",
              "afterDriveFront.png", "afterDriveBack.png", "afterPassengerFront.png",
              "afterPassengerBack.png"]

db_column = ["", "before_front", "before_back", "before_drive_front", "before_drive_back",
             "before_passenger_front", "before_passenger_back", "after_front", "after_back",
             "after_drive_front", "after_drive_back", "after_passenger_front", "after_passenger_back"]
db_column_s = ["", "before_front_count", "before_back_count", "before_drive_front_count", "before_drive_back_count",
             "before_passenger_front_count", "before_passenger_back_count", "after_front_count", "after_back_count",
             "after_drive_front_count", "after_drive_back_count", "after_passenger_front_count", "after_passenger_back_count"]


def load_data():
    project_car = pymysql.connect(
        host='localhost',
        port=3306,
        user='root',
        password='003674',
        db='project_car',
        charset='utf8',
        local_infile=1
    )
    print("Database Connect")
    cursor = project_car.cursor(pymysql.cursors.DictCursor)

    # rent테이블 중 rent_status가 6인 것 만 로드
    sql = "SELECT rent_id FROM rent where rent_status=6;"
    cursor.execute(sql)
    results_id = cursor.fetchall()
    print("Rent_id 가 6인 results :", results_id)  # list

    # 만약 rent_id가 없으면 실행 중지
    if not results_id:
        print("nothing Rent_status == 6")
        return 0 # main while루프로 0 값 전달
    else:
        # 3과 6인 rent id 저장
        rent_ids = []
        for i in range(len(results_id)):
            print(results_id[i]['rent_id'])
            rent_ids.append(results_id[i]['rent_id'])
        print(rent_ids)

    print("load_data success")
    project_car.close()  # 연결 닫기
    print("Database Connect End")
    return rent_ids


def match_rent_id():
    print(len(rent_ids_lists))
    # rent_ids_lists 를 가져왔음 그건 딥러닝 수행 전 데이터임
    for i in range(len(rent_ids_lists)):
        if (i == len(rent_ids_lists)):
            break
        else:
            print("i, rent_ids_lists :", i, rent_ids_lists[i])
            # 딥러닝 수행 및 데베 업데이트
            update_data(rent_ids_lists[i])


def update_data(rent_id_data):
    project_car = pymysql.connect(
        host='localhost',
        port=3306,
        user='root',
        password='003674',
        db='project_car',
        charset='utf8',
        local_infile=1
    )
    print("Database Connect")
    cursor = project_car.cursor(pymysql.cursors.DictCursor)  # 디폴트 커서 생성
    print(type(rent_id_data))
    
    # rent-id가 같은거 찾기
    sql = "select * from rent_compare_img where Rent_rent_id = " + str(rent_id_data) + ";"
    print("rent_compare_img find sql = ", sql)
    cursor.execute(sql)
    results_data = cursor.fetchall()
    print("results_data :", results_data)

    for data in results_data:
        # print("rent_id_data :", key, values)
        print("rent_id_data :", data)
        print("type :", type(data))
        db_column_idx = 0
        for key, values in data.items():
            if db_column_idx < 1:
                print("items :", db_column_idx, key, values)
                db_column_idx = db_column_idx + 1
                continue
            else:
                print("items :", db_column_idx, key, values)
                # 여기서 컬럼 이름 화긴 후 detect or pass 수행
                save_dir_img_path = obj_detect(values, db_column_idx, rent_id_data)
                print("sava_dir_img : ", save_dir_img_path)
                # 데이터베이스에 업데이트
                # 이미지 path
                sql = "update rent_compare_img set " + db_column[db_column_idx] + " = '" + save_dir_img_path[0] + "' where Rent_rent_id = " + str(rent_id_data) + ";"
                print(sql)
                cursor.execute(sql)
                # # 스크래치 갯수
                sql2 = "update rent_scratch_count set " + db_column_s[db_column_idx] + " = '" + str(save_dir_img_path[1]) + "' where rent_id = " + str(rent_id_data) + ";"
                print(sql2)
                cursor.execute(sql2)
                project_car.commit()
                db_column_idx = db_column_idx + 1
                if db_column_idx == 13:
                    break

    project_car.close()  # 연결 닫기
    print("Database Connect End")
    print("update_data success")
    print("#################################################################################################")


def obj_detect(values, db_column_idx, rent_id_data):
    file_name = values
    weights_file = "custom-train-yolo3_best.weights"
    cfg_file = "custom-train-yolo3.cfg"
    classes_names_file = "classes.names"

    # 딥러닝 시작
    # rent id 의 폴더에 접근하기
    # detect 된 파일은 저장하고 데베에 덮어씌우기

    # @ 시작 시간 / 걸린 시간 체크하기
    start_time = time.time()

    # ! Yolo Load
    net = cv2.dnn.readNetFromDarknet(cfg_file, weights_file)  # cfg 파일이 먼저

    # ! 클래스(coco.names) 파일 열기
    with open(classes_names_file, "r") as f:
        classes = [line.strip() for line in f.readlines()]

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

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

    # ! 노이즈 제거 (중복 박스 제거) -> 박스 클래스
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, min_confidence, nms_threshold)
    
    # 스크래치 갯수 세는 인덱스
    scratch_num = 0

    # ! 화면에 표시하기 위한 코드
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]  # 좌측 x, 상단 y, 우측 x, 하단 y 좌표
            # object_lable = "{} : {:0.4f}".format(classes[class_ids[i]], confidences[i])
            object_lable = "{}".format(classes[class_ids[i]])
            color = (255,255,0)
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, object_lable, (x, y), font, 2, color, 2)
            scratch_num = scratch_num + 1

    print("scratch_num : ", scratch_num)

    # @ 걸린 시간 계산
    end_time = time.time()
    process_time = end_time - start_time
    print("=== A frame took {:.3f} seconds".format(process_time))

    # ! 이미지 출력 저장
    # cv2.imshow("Detected Img", img)
    # save_dir_img_path = obj_detect_dir_path + name_lists[db_column_idx]
    save_dir_img_path = obj_detect_dir_path + str(rent_id_data) + "/" + name_lists[db_column_idx]
    cv2.imwrite(save_dir_img_path, img)

    return [save_dir_img_path, scratch_num]

#########################################################################################################

# main 실행
while True:
    print("######## main 실행 #######")
    print("rent 테이블 데이터 로드")
    print("load_data")
    # rent 테이블에서 ststus = 6인 rent_id 리스트 형식으로 꺼내오기
    rent_ids_lists = load_data()
    print("main :" , rent_ids_lists)
    if not rent_ids_lists:
        continue
    else:
        # rent 테이블과 rent_compare_img 테이블에서 rent_id가 같은거 찾아서 update_data()실행
        match_rent_id()
        print("Delay 10s")
        # time.sleep(20)

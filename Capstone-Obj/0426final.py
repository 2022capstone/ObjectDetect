import pymysql
# import pandas as pd
import cv2
import numpy as np
import time
import os

obj_detect_dir_path = "/Users/jongukyang/Capstone-Obj/dmig/"

name_lists = ["", "d_before_front.png", "d_before_back.png", "d_before_drive_front.png", "d_before_drive_back.png",
              "d_before_passenger_front.png", "d_before_passenger_back.png", "d_after_front.png",
              "d_after_back.png",
              "d_after_drive_front.png", "d_after_drive_back.png", "d_after_passenger_front.png",
              "d_after_passenger_back.png"]

db_column = ["", "before_front", "before_back", "before_drive_front", "before_drive_back",
             "before_passenger_front", "before_passenger_back", "after_front", "after_back",
             "after_drive_front", "after_drive_back", "after_passenger_front", "after_passenger_back"]


def load_data():
    project_car = pymysql.connect(
        host='localhost',
        port=3306,
        user='root',
        password='whddnr15',
        db='project_car',
        charset='utf8',
        local_infile=1
    )
    print("Database Connect")
    cursor = project_car.cursor(pymysql.cursors.DictCursor)

    # rent테이블 전체 로드
    sql = "select * from rent"
    cursor.execute(sql)
    results = cursor.fetchall()
    print("Main results :", results)  # 딕셔너리

    # 데베 테이블 항목 전체 개수 확인
    sql2 = "select count(*) from rent"
    cursor.execute(sql2)
    results2 = cursor.fetchall()
    # print(type(results2[0]['count(*)']))
    print("총 개수 : ", results2[0]['count(*)'])  # int형

    # rent_id 빼내기
    rent_ids = []
    for i in range(results2[0]['count(*)']):
        rent_id = results[i]['rent_id']
        rent_ids.append(rent_id)
    print("rent_ids =", rent_ids)

    # results = pd.DataFrame(results)
    # print(results)
    print("load_data success")
    project_car.close()  # 연결 닫기
    print("Database Connect End")
    return rent_ids


def load_last_rent_id_data():
    # rent_ids = load_data()
    project_car = pymysql.connect(
        host='localhost',
        port=3306,
        user='root',
        password='whddnr15',
        db='project_car',
        charset='utf8',
        local_infile=1
    )
    print("Database Connect")
    cursor = project_car.cursor(pymysql.cursors.DictCursor)
    # 마지막 rent_id  img들 딥러닝 돌리기
    cnt = 0
    while True:
        if (cnt == len(rent_ids_lists)):
            print("cnt = ", cnt)
            r_id = rent_ids_lists[cnt - 1]
            break
        cnt = cnt + 1
    # print(r_id)
    sql = "select * from rent_compare_img where Rent_rent_id = " + str(r_id) + ";"
    print(sql)
    cursor.execute(sql)
    results = cursor.fetchall()
    print("Main results :", results)  # 딕셔너리
    print("load_rent_id_data success")
    project_car.close()  # 연결 닫기
    print("Database Connect End")
    return results


def update_data(rent_id_data):
    project_car = pymysql.connect(
        host='localhost',
        port=3306,
        user='root',
        password='whddnr15',
        db='project_car',
        charset='utf8',
        local_infile=1
    )
    print("Database Connect")
    cursor = project_car.cursor(pymysql.cursors.DictCursor)  # 디폴트 커서 생성

    print("rent_id_data (마지막 id) :", rent_id_data[0]['Rent_rent_id'])
    print("rent_id_data (마지막 id) :", type(rent_id_data))

    last_rent_id = rent_id_data[0]['Rent_rent_id']

    # 마지막 rent_id  img들 딥러닝 돌리기

    for data in rent_id_data:
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

                save_dir_img = obj_detect(values, db_column_idx)

                print("sava_dir_img : ", save_dir_img)

                sql = "update rent_compare_img set " + db_column[db_column_idx] + " = '" + save_dir_img + "' where Rent_rent_id = " + str(last_rent_id) + ";"
                print(sql)
                cursor.execute(sql)
                project_car.commit()

                db_column_idx = db_column_idx + 1
                if db_column_idx == 13:
                    break

    project_car.close()  # 연결 닫기
    print("Database Connect End")
    print("update_data success")


# def save_data(rent_ids_lists):
#     cnt = 0
#     while True:
#         if (cnt == len(rent_ids_lists)):
#             print("cnt = ", cnt)
#             r_id = rent_ids_lists[cnt-1]
#             break
#         cnt = cnt + 1
#
#     print(r_id)
#
#     project_car = pymysql.connect(
#         host='localhost',
#         port=3306,
#         user='root',
#         password='whddnr15',
#         db='project_car',
#         charset='utf8',
#         local_infile=1
#     )
#     print("Database Connect")
#     cursor = project_car.cursor(pymysql.cursors.DictCursor)# 디폴트 커서 생성
#     # 전달받은 고객의 rent_id 확인 후 저장
#     # sql2 = "select imgurl from carimg where id=" + str(rent_id) + ";"
#
#     sql = "INSERT INTO rent_compare_img( Rent_rent_id, before_front, before_back, before_drive_front, " \
#           "before_drive_back, before_passenger_front, before_passenger_back, after_front, after_back, " \
#           "after_drive_front, after_drive_back, after_passenger_front, after_passenger_back) " \
#           "VALUES(" + str(r_id) + ", " \
#           "'/Users/jongukyang/Capstone-Obj/dmig/', " \
#           "'/Users/jongukyang/Capstone-Obj/dmig', " \
#           "'/Users/jongukyang/Capstone-Obj/dmig', " \
#           "'/Users/jongukyang/Capstone-Obj/dmig', " \
#           "'/Users/jongukyang/Capstone-Obj/dmig', " \
#           "'/Users/jongukyang/Capstone-Obj/dmig', " \
#           "'/Users/jongukyang/Capstone-Obj/dmig', " \
#           "'/Users/jongukyang/Capstone-Obj/dmig', " \
#           "'/Users/jongukyang/Capstone-Obj/dmig', " \
#           "'/Users/jongukyang/Capstone-Obj/dmig', " \
#           "'/Users/jongukyang/Capstone-Obj/dmig', " \
#           "'/Users/jongukyang/Capstone-Obj/dmig');"
#
#     cursor.execute(sql)
#     project_car.commit()
#     # print('rowcount: ', testpython.rowcount)
#     project_car.close()  # 연결 닫기
#     print("Database Connect End")
#     print("save_data success")

def obj_detect(values, db_column_idx):
    file_name = values
    weights_file = "yolov3.weights"
    cfg_file = "yolov3.cfg"
    classes_names_file = "coco.names"

    # 딥러닝 시작
    # rent id 의 폴더에 접근하기
    # detect 된 파일은 d_ 이름을 붙여서 저장하고 데베에 덮어씌우기

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

    save_dir_img = obj_detect_dir_path + name_lists[db_column_idx]
    cv2.imwrite(save_dir_img, img)

    return save_dir_img
    # cv2.waitKey(0)  # 키 입력을 기다리는 함수
    # cv2.destroyAllWindows()  # 키 입력되면 창 닫기 종료
    # update_data()


#########################################################################################################

# main 실행

while True:
    print("######## main 실행 #######")
    # print("save_data start")
    # Rent_rent_id = 7
    # print(type(Rent_rent_id))
    # save_data(Rent_rent_id)
    print("rent 테이블 데이터 로드")
    print("load_data")
    rent_ids_lists = load_data()  # rent id list 전체
    print(rent_ids_lists)
    rent_id_data = load_last_rent_id_data()  # 마지막 rent id 번호
    # print(type(rent_ids_lists[0]))
    # print("rent_ids를 가져와 마지막 인덱스 골라 save 실행")
    # save_data(rent_ids_lists)
    print("rent_id 를 통해 update_data 실행 및 딥러닝")
    update_data(rent_id_data)
    print("Delay 10s")
    time.sleep(10)

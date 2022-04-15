import pymysql
# import pandas as pd
import cv2
import numpy as np
import time


def load_data():
    project_car = pymysql.connect(
        host='localhost',
        port=3306,
        user='root',
        password='1234',
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
    print("Main results :", results) # 딕셔너리

    # 데베 테이블 항목 전체 개수 확인
    sql2 = "select count(*) from rent"
    cursor.execute(sql2)
    results2 = cursor.fetchall()
    # print(type(results2[0]['count(*)']))
    print("총 개수 : ", results2[0]['count(*)']) # int형

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

def update_data(rent_ids_list):
    # rent_ids 리스트 내에 rent_id 들이 있고 그걸 어떻게 사용?
    # 모든 데베에 접근해서 계속 딥러닝 돌리기
    # while문? 아무튼 딥러닝된 파일 이름을 대조해보면서 반복하기
    # 딜레이는 1분 내외로 설정하기

    # 딥러닝 시작
    # rent id 의 폴더에 접근하기
    # detect 된 파일은 d_ 이름을 붙여서 저장하고 데베에 덮어씌우기

    name_list = ["d_before_front.jpg", "d_before_back.jpg", "d_before_drive_front.jpg", "d_before_drive_back.jpg",
                 "d_before_passenger_front.jpg", "d_before_passenger_back.jpg", "d_after_front.jpg", "d_after_back.jpg",
                 "d_after_drive_front.jpg", "d_after_drive_back.jpg", "d_after_passenger_front.jpg",
                 "d_after_passenger_back.jpg"]

    # 전체 데베 탐색은 load data를 반복 돌립시다 while True:
    for count in rent_ids_list:
        print("count rent_id :", count)
        # 데베에 사진 접근

    # for result in results:
    #     print("result :",result)
    #     if rent_id == result['id']:
    #         # 딥러닝
    #         # 사진 url 로 접근하기
    #         print("i am 10")
    #         for key, values in result.items():
    #             # print("data :", key, values)
    #             if values != str(values): # 문자열인 경우에만 가져와야함 그걸 이렇게 쓰는게 맞나
    #                 continue
    #             elif values == str(values):
    #                 print("values :", values)
    #                 url = values
    #                 print("start obj detect !!!")
    #                 object_detection(url, cursor, testpython, name_list[count], rent_id)
    #                 count = count+1
    #             else:
    #                 break


def save_data(Rent_rent_id):
    project_car = pymysql.connect(
        host='localhost',
        port=3306,
        user='root',
        password='1234',
        db='project_car',
        charset='utf8',
        local_infile=1
    )
    print("Database Connect")
    cursor = project_car.cursor(pymysql.cursors.DictCursor)# 디폴트 커서 생성
    # 전달받은 고객의 rent_id 확인 후 저장
    # sql2 = "select imgurl from carimg where id=" + str(rent_id) + ";"

    sql = "INSERT INTO rent_compare_img( Rent_rent_id, before_front, before_back, before_drive_front, " \
          "before_drive_back, before_passenger_front, before_passenger_back, after_front, after_back, " \
          "after_drive_front, after_drive_back, after_passenger_front, after_passenger_back) " \
          "VALUES(" + str(Rent_rent_id) + ", " \
          "'C:\\Users\\User\\Desktop\\ObjectDetect-main\\Capstone-Obj\\dimg\\jongukposter.jpeg', " \
          "'C:\\Users\\User\\Desktop\\ObjectDetect-main\\Capstone-Obj\\dimg\\jongukposter.jpeg', " \
          "'C:\\Users\\User\\Desktop\\ObjectDetect-main\\Capstone-Obj\\dimg\\jongukposter.jpeg', " \
          "'C:\\Users\\User\\Desktop\\ObjectDetect-main\\Capstone-Obj\\dimg\\jongukposter.jpeg', " \
          "'C:\\Users\\User\\Desktop\\ObjectDetect-main\\Capstone-Obj\\dimg\\jongukposter.jpeg', " \
          "'C:\\Users\\User\\Desktop\\ObjectDetect-main\\Capstone-Obj\\dimg\\jongukposter.jpeg', " \
          "'C:\\Users\\User\\Desktop\\ObjectDetect-main\\Capstone-Obj\\dimg\\jongukposter.jpeg', " \
          "'C:\\Users\\User\\Desktop\\ObjectDetect-main\\Capstone-Obj\\dimg\\jongukposter.jpeg', " \
          "'C:\\Users\\User\\Desktop\\ObjectDetect-main\\Capstone-Obj\\dimg\\jongukposter.jpeg', " \
          "'C:\\Users\\User\\Desktop\\ObjectDetect-main\\Capstone-Obj\\dimg\\jongukposter.jpeg', " \
          "'C:\\Users\\User\\Desktop\\ObjectDetect-main\\Capstone-Obj\\dimg\\jongukposter.jpeg', " \
          "'C:\\Users\\User\\Desktop\\ObjectDetect-main\\Capstone-Obj\\dimg\\jongukposter.jpeg');"

    cursor.execute(sql)
    project_car.commit()
    # print('rowcount: ', testpython.rowcount)
    project_car.close()  # 연결 닫기
    print("Database Connect End")
    print("save_data success")

#########################################################################################################
# main 실행
print("######## main 실행 #######")
# Rent_rent_id = 6
# print(type(Rent_rent_id))
# save_data(Rent_rent_id)
print("rent 테이블 데이터 로드")
load_data()
rent_ids_list = load_data()
print(type(rent_ids_list[0]))
print("rent_ids 를 통해 update_data 실행 및 딥러닝")
update_data(rent_ids_list)
import pymysql
import pandas as pd

# 데이터베이스에 저장된 이미지를 가져온다
# 이미지는 for문과 sql 문의 조합으로 가져와야할듯?
# 가져온 이미지는 저장 또는 전송 필요
# 이미지를 object detect하기 위해 read 시켜야함

def load_data():
    testpython = pymysql.connect(
        host='localhost',
        port=3306,
        user='root',
        password='whddnr15',
        db='pycharm',
        charset='utf8',
        local_infile=1
    )
    print("Database Connect")
    # 연결한 DB와 상호작용하기 위해 cursor 객체를 생성해주어야 합니다.
    # 다양한 커서의 종류가 있지만,
    # Python에서 데이터 분석을 주로 pandas로 하고
    # RDBMS(Relational Database System)를 주로 사용하기 때문에
    # 데이터 분석가에게 익숙한 데이터프레임 형태로 결과를 쉽게 변환할 수 있도록
    # 딕셔너리 형태로 결과를 반환해주는 DictCursor를 사용하겠습니다.
    cursor = testpython.cursor(pymysql.cursors.DictCursor)

    # sql = "select * from carimg;"
    # 전달받은 고객의 rent_id 확인 후 url 불러오기
    sql = "select * from carimg;"
    cursor.execute(sql)
    result = cursor.fetchall()
    print(result)
    result = pd.DataFrame(result)
    print(result)
    print("load_data success")
    testpython.close()  # 연결 닫기
    print("Database Connect End")

def save_data():
    testpython = pymysql.connect(
        host='localhost',
        port=3306,
        user='root',
        password='whddnr15',
        db='pycharm',
        charset='utf8',
        local_infile=1 ) # DB 연동
    cursor = testpython.cursor() # 디폴트 커서 생성
    # 전달받은 고객의 rent_id 확인 후 저장
    sql = "INSERT INTO carimg(id, imgurl, imgurl2, imgurl3) VALUES(15, '/Users/jongukyang/Desktop/hello.jpeg', 'desktop/hello2', 'desktop/hello3');"
    cursor.execute(sql)
    testpython.commit()
    # print('rowcount: ', testpython.rowcount)
    testpython.close()  # 연결 닫기
    print("save_data success")


save_data()



load_data()
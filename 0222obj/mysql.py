import pymysql
import pandas as pd


testpython=pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='whddnr15',
    db='pycharm',
    charset='utf8',
    local_infile=1
)

# 연결한 DB와 상호작용하기 위해 cursor 객체를 생성해주어야 합니다.
# 다양한 커서의 종류가 있지만,
# Python에서 데이터 분석을 주로 pandas로 하고
# RDBMS(Relational Database System)를 주로 사용하기 때문에
# 데이터 분석가에게 익숙한 데이터프레임 형태로 결과를 쉽게 변환할 수 있도록
# 딕셔너리 형태로 결과를 반환해주는 DictCursor를 사용하겠습니다.
cursor = testpython.cursor(pymysql.cursors.DictCursor)

sql = "select * from `carimg`;"
cursor.execute(sql)
result = cursor.fetchall()

result = pd.DataFrame(result)
print(result)






import psycopg2
import openpyxl

import os
import sys

# root 폴더 설정 (nlp_deeplearning/)
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from config.DatabaseConfig import *

# 기존 학습 데이터 초기화
def all_clear_train_data(db):
  sql = "DELETE FROM chatbot_train"

  with db.cursor() as cursor:
    cursor.execute(sql)
  db.commit()

  # auto increment 초기화
  sql = "ALTER SEQUENCE chatbot_train_id_seq RESTART WITH 1"

  with db.cursor() as cursor:
    cursor.execute(sql)
  db.commit()

# DB에 데이터 저장
def insert_data(db, xls_row):
  # 셀 객체에서 값을 추출
  intent = xls_row[0].value
  ner = xls_row[1].value
  query = xls_row[2].value
  answer = xls_row[3].value
  answer_img_url = xls_row[4].value

  sql = "INSERT INTO chatbot_train(intent, ner, query, answer, answer_image) VALUES (%s, %s, %s, %s, %s)"

  with db.cursor() as cursor:
    cursor.execute(sql, (intent, ner, query, answer, answer_img_url))
  db.commit()

# 합습 엑셀 파일 경로
train_file = os.path.join(root_dir, "train_tools", "train_data.xlsx")

db = None

try:
  
  db = psycopg2.connect(
    host=DB_HOST,
    database=DB_NAME,
    user=DB_USER,
    password=DB_PASS,
    port=DB_PORT
  )

  # 기존 학습 데이터 초기화
  all_clear_train_data(db)

  # 학습 파일 호출
  wb = openpyxl.load_workbook(train_file)
  sheet = wb['Sheet1']

  for row in sheet.iter_rows(min_row=2): # 헤더 부분을 제외
    # 데이터 저장
    insert_data(db, row)
except Exception as e:
  print(e)
finally:
  if db is not None:
    db.close()
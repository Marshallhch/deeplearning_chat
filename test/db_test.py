import os, sys
from pathlib import Path
import tensorflow as tf

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from dotenv import load_dotenv
load_dotenv()

from config.connect_database import Database
from utils.preprocess import Preprocess
from model.intent.intent_model import IntenModel
from model.ner.ner_model import NerModel
from utils.find_answer import FindAnswer
from config.DatabaseConfig import * 

# 전처리 객체 생성 -> 데이터베이스 연결 -> 의도 파악(모델 활용) -> 개체명 인식(모델 활용) -> 답변 검색

# 전처리 객체 생성
p = Preprocess(word2index_dic=os.path.join(root_dir, 'train_tools', 'dict', 'chatbot_dict.bin'), userdic=os.path.join(root_dir, 'test', 'user_dic.tsv'))

# 데이터베이스 연결
db = Database(host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME, port=DB_PORT)

# 샘플 테스트 문장
query = '내일 오전 10시에 탕수육 배달 예약할께요'
# query = '안녕하세요. 반갑습니다.'

# 커스텀 오브젝트 정의
custom_objects = {'softmax_v2': tf.nn.softmax}
intent = IntenModel(model_name=os.path.join(root_dir, 'model', 'intent', 'intent_model.keras'), preprocess=p, custom_objects=custom_objects)

predict = intent.predict_class(query)
intent_name = intent.labels[predict]

# print('의도 파악: ', {intent_name})

# 개체명 인식
ner = NerModel(model_name=os.path.join(root_dir, 'model', 'ner', 'ner_model.keras'), preprocess=p)
predicts = ner.predict(query)
ner_tags = ner.predict_tags(query)

# print('개체명 인식: ', predicts)
# print('답변 검색에 필요한 NER 태그: ', ner_tags)

# 답변 검색
try:
  f = FindAnswer(db)
  answer_text, answer_image = f.search(intent_name, ner_tags)
  answer = f.tag_to_word(predicts, answer_text)
except:
  answer = '죄송합니다. 좀 더 공부할께요!'

print('답변: ', {answer})

try:
  # 데이터베이스가 연결되어 있거나 None 값이 아닐때
  if hasattr(db, 'conn') and db.conn is not None:
    db.close()
except Exception as e:
  print('데이터베이스 연결 종료 중 오류: ', e)
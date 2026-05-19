import os, sys
from pathlib import Path

import tensorflow as tf
import threading # 멀티 스레드 사용을 위한 모듈
import json

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, root_dir)

from dotenv import load_dotenv
load_dotenv()

from utils.preprocess import Preprocess
from utils.bot_server import BotServer
from model.intent.intent_model import IntenModel
from model.ner.ner_model import NerModel
from config.connect_database import Database
from utils.find_answer import FindAnswer

from config.DatabaseConfig import *

# 업그레이드 소프트맥스 저장
custom_objects = {'softmax_v2': tf.nn.softmax}

# 전처리 객체 생성 -> 의도 분류 모델 객체 생성 -> 개체명 인식 -> 클라이언트의 서버 연결 -> 데이터베이스 연결 -> 클라이언트 데이터 수신 -> 의도 파악 및 개체명 파악 -> 답변 검색 -> 답변을 json 형식으로 반환 -> 서버 실행(포트, 최대 연결 수 지정) -> 스레드 시작

# 전처리 객체 생성
p = Preprocess(word2index_dic=os.path.join(root_dir, 'train_tools', 'dict', 'chatbot_dict.bin'), userdic=os.path.join(root_dir, 'test', 'user_dic.tsv'))

# 의도 분류 모델 객체 생성
intent = IntenModel(model_name=os.path.join(root_dir, 'model', 'intent', 'intent_model.keras'), preprocess=p, custom_objects=custom_objects)

# 개체명 인식
ner = NerModel(model_name=os.path.join(root_dir, 'model', 'ner', 'ner_model.keras'), preprocess=p)

# 클라이언트 서버 연결
# 클라이언트와 서버 연결이 수락되는 순간 실행되는 함수
# 질의한 내용을 처리해 답변을 찾은 후 클라이언트에 응답을 전송
def to_client(conn, addr):
  # 각 스레드마다 독립적인 데이터 베이스 연결 생성
  db = Database(host=DB_HOST, user=DB_USER, password=DB_PASS, database=DB_NAME, port=DB_PORT)

  try:
    # 데이터 수신
		# conn은 클라이언트와 연결된 소켓 객체 - 이를 통해 데이터를 클라이언트와 주고 받는다
		# recv() 메서드는 데이터가 수신될 때까지 브로킹 상태로 대기
		# 최대 2048 바이트 만큼 데이터 수신
		# 클라이언트와 연결이 끊어지거나 오류가 있는 경우에는 브로킹 해제되고 None 반환
    read = conn.recv(2048)
    print('=' * 50)
    print('connection from: %s' % str(addr))

    if read is None or not read:
      print('클라이언트 연결 끊어짐')

    # 수신된 read 데이터를 json 형식으로 변화
    recv_json_data = json.loads(read.decode())

    print('데이터 수신: ', recv_json_data) # Query 키가 적용되어 있음
    query = recv_json_data['Query'] # 질의 내용

    # 의도 파악 디버깅 출력
    print('입력 질문: ', query)
    print('질문 타입: ', type(query))

    # 의도 파악
    intent_predict = intent.predict_class(query)
    intent_name = intent.labels[intent_predict]

    print('의도 예측: ', intent_name) # '주문'

    # 개체명 예측
    ner_predict = ner.predict(query)
    ner_tags = ner.predict_tags(ner_predict)

    print('개체명 예측: ', ner_tags) # 'B_FOOD'

    # 데이터베이스에 답변 검색
    try:
      f = FindAnswer(db)
      answer_text, answer_image = f.search(intent_name, ner_tags)
      answer = f.tag_to_word(ner_predict, answer_text) # 'B_FOOD' -> '탕수육'
    except:
      answer = "죄송합니다. 문제가 발생했습니다. 잠시후 다시 이용해 주세요."
      answer_image = None

    # 개체명 답변을 리스트 형식으로 변환
    ner_predict_list = []
    for pred in ner_predict:
      if isinstance(pred, tuple):
        ner_predict_list.append(list(pred))
      else:
        ner_predict_list.append(pred)

    # 답변을 json 형식으로 변환
    send_json_data = {
      "Query": query,
      'Answer': answer,
      'AnswerImageUrl': answer_image,
      'Intent': intent_name,
      'NER': ner_predict_list
    }

    message = json.dumps(send_json_data)
    conn.send(message.encode())
  except Exception as ex:
    print('메시지 전송 오류: ', ex)

  except Exception as e:
    print('클라이언트 서버 요청 오류: ', e)
  finally:
    if db is not None:
      db.close()
    conn.close()

if __name__ == '__main__':
  # 서버 동작
  port = 5050
  listen = 100 # 최대 연결 수
  bot = BotServer(port, listen)
  bot.create_sock() # 소캣 생성
  print('봇 서버 실행 완료!')

  # 무한루프를 돌면서 챗봇 클라이언트 연결을 기다린다.
	# 클라이언트 서버 요청이 서버에서 수락되는 즉시 챗봇 서비스 요청을 처리하는 스레드 생성
	# 생성되는 스레드는 to_client() 함수를 호출하여 서비스 요청 처리
	# 각 스레드에서 독립적인 데이터베이스 연결을 생성합니다.
  while True:
    conn, addr = bot.ready_for_client() # 클라이언트 대기
    client = threading.Thread(target=to_client, args=(conn, addr))
    client.start() # 스레드 시작
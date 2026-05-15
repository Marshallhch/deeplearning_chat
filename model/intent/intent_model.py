import sys
import os
from dotenv import load_dotenv
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import preprocessing

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

load_dotenv()

# 소프트맥스 업그레이드 버전 사용
custom_objects = {'softmax_v2': tf.nn.softmax}

# 의도 분류 모듈 클래스
class IntenModel:
  def __init__(self, model_name, preprocess, custom_objects=None):
    # 의도 클래스 레이블
    self.labels = {0: '인사', 1: '욕설', 2: '주문', 3: '예약', 4: '기타'}
    # 의도 분류 모델 호출
    self.model = load_model(model_name, custom_objects=custom_objects)
    # 전처리 객체
    self.p = preprocess

  # 의도 클래스 예측
  def predict_class(self, query):
    # 입력값 문자열 처리
    if isinstance(query, list):
      query = " ".join(map(str, query))
    query = str(query) # 테스트 시 리스트 형태는 오류가 난다. 따라서 문자열로 반환해 준다

    # 디버깅 출력 - 테스트 오류 시 사용
    # print('처리된 질문: ', query)
    # print('질문 타입: ', type(query))

    # 형태소 분석: [('처리된', 'Verb'), ("질문", 'Noun')]
    pos = self.p.pos(query)

    # print(pos)

    # 불용어 제거
    keywords = self.p.get_keywords(pos, without_tag=True) # ('타입', 'Noun') -> ['타입']
    sequences = [self.p.get_wordidx_sentence(keywords)] # ['타입'] -> [117]

    # 패딩 처리
    padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=15, padding='post')

    # 입력한 문장 예측
    predict = self.model.predict(padded_seqs)
    predict_class = tf.math.argmax(predict, axis=1)
    return predict_class.numpy()[0]
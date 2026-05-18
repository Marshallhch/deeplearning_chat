import tensorflow as tf
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras import preprocessing

# 개체명 인식 클래스
class NerModel:
  def __init__(self, model_name, preprocess):
    # BIO 태그 클래스 별 레이블
    self.index_to_ner = {1: 'O', 2: 'B_DT', 3: 'B_FOOD', 4: 'I', 5: 'B_OG', 6: 'B_PS', 7: 'B_LC', 8: 'NNP', 9: 'B_TI', 0: 'PAD'}
    # 의도 분류 모델 호출
    self.model = load_model(model_name)
    # 전처리 객체
    self.p = preprocess

    # 형태소 분석 -> 문장 내 키워드 추출 및 불용어 제거(get_keywords, get_wordidx_sentence) -> 패딩 처리 pad_sequences -> 키워드 별 개체명 예측

  def predict(self, query):
    # 형태소
    pos = self.p.pos(query)

    # 키워드 추출 및 불용어 제거
    keywords = self.p.get_keywords(pos, without_tag=True)
    sentences = [self.p.get_wordidx_sentence(keywords)]

    # 패딩
    max_len = 40
    padded_seqs = preprocessing.sequence.pad_sequences(sentences, padding='post', maxlen=max_len, value=0) # value = 패딩값

    # 키워드 별 개체명 예측
    predicts = self.model.predict(np.array([padded_seqs[0]]))
    predict_class = tf.math.argmax(predicts, axis=-1) # 가장 높은 확률을 가진 클래스 선택

    tags = [self.index_to_ner[i] for i in predict_class.numpy()[0]]
    return list(zip(keywords, tags))

  # 태그 예측
  def predict_tags(self, query):
    # 문자열 입력 받아 형태소 분석
    if isinstance(query, str): # 문자열로 받아질 경우
      # 형태소 분석
      pos = self.p.pos(query)
      # 문장 내 키워드 추출 및 불용어 제거
      keywords = self.p.get_keywords(pos, without_tag=True)
    else:
      # predict() 메서드의 결과를 받았을 겨우
      keywords = [word for word in query]
    sentences = [self.p.get_wordidx_sentence(keywords)]

    # 패딩
    max_len = 40
    padded_seqs = preprocessing.sequence.pad_sequences(sentences, padding='post', maxlen=max_len, value=0)

    # 키워드 별 개체명 예측
    predicts = self.model.predict(np.array([padded_seqs[0]]))
    predict_class = tf.math.argmax(predicts, axis=-1)

    # 태그 추출
    tags = []
    for tag_idx in predict_class.numpy()[0]:
      if tag_idx == 1: continue
      tags.append(self.index_to_ner[tag_idx])

    if len(tags) == 0: return None
    return tags

  
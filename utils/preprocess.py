from dotenv import load_dotenv
from konlpy.tag import Komoran
import pickle
import os

load_dotenv()

# komoram 설치 확인
# print(Komoran().morphs('내일 삼성전자 주가는 얼마나 오를까?'))

# 전처리 클래스
class Preprocess:
  """
  word2index_dic: 단어 인덱스 사전 파일 경로
  userdic: 사용자 정의 사전 파일 경로
  """

  def __init__(self, word2index_dic="", userdic=None):
    # 단어 인덱스 사전 추가
    if word2index_dic != "":
      f = open(word2index_dic, 'rb')
      self.word_index = pickle.load(f)
      f.close()
    else:
      self.word_index = None

    # 사용자 사전 파일이 실제로 존재할 때만 적용
    if userdic is not None and not os.path.exists(userdic):
      userdic = None

    # 형태소 분석기 초기화
    self.komoran = Komoran(userdic=userdic)
    
    # 제외할 품사
		# 참조: https://docs.komoran.kr/firststep/postypes.html
		# 관계언 제거
		# 어미 제거
		# 접미사 제거
    self.exclusion_tags = [
			'JKS', 'JKQ', 'JKV', 'JKC', 'JKG', 'JKO', 'JKB', 'JKS', 'JX', 'JC',
			'SF', 'SP', 'SS', 'SE', 'SO',
			'EP', 'EF', 'EC', 'ETN', 'ETM',
			'XSN', 'XSV', 'XSA'
		]

  # 형태소 분석기 POS 태그 추출
  def pos(self, sentence):
    # 입력값 문자열 처리
    if isinstance(sentence, list):
      sentence = " ".join(sentence)
    sentence = str(sentence)

    # print('형태소 분석 입력: ', sentence)

    return self.komoran.pos(sentence)
  
  # 불용어 제거 후 필요한 정보만 추출
  def get_keywords(self, pos, without_tag=False):
    # self.exclusion_tags 리스트에 포함돼 있는 품사 정보만 키워드로 저장
    f = lambda x: x in self.exclusion_tags
    word_list = []

    for p in pos:
      if f(p[1]) is False:
        # ('오전', 'NNP'): without_tag=False - 단어만 추출
        word_list.append(p if without_tag is False else p[0])
    return word_list

  # 키워드를 단어 인덱스 시퀀스로 변환
  def get_wordidx_sentence(self, keywards):
    if self.word_index is None:
      return []
    
    w2i = []
    for word in keywards:
      try:
        w2i.append(self.word_index[word])
      except KeyError:
        w2i.append(self.word_index["OOV"])
    return w2i
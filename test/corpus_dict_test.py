import sys
import os
import pickle

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from dotenv import load_dotenv
load_dotenv()

from utils.preprocess import Preprocess

# 단어 사전 호출
f = open(os.path.join(root_dir, 'train_tools', 'dict', 'chatbot_dict.bin'), 'rb')
word_index = pickle.load(f)
f.close()

# 예시문장
sent = '내일 오전 10시에 탕수육 주문하고 싶어요.'

# 사용자 사전 파일 호출
user_dict_path = os.path.join(root_dir, "test", "user_dict.tsv")

# 전처리 객체 생성
p = Preprocess(userdic=user_dict_path)

# 형태소 분석기
pos = p.pos(sent)

# 품사 태그 없는 키워드 출력
keywords = p.get_keywords(pos, without_tag=True)

for word in keywords:
  try:
    print(word, word_index[word])
  except KeyError:
    print(word, word_index['OOV'])
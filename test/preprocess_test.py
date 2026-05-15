import sys
import os

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

from dotenv import load_dotenv
load_dotenv()

# 사용자 사전 파일 저장
user_dic_path = os.path.join(root_dir, "test", "user_dic.tsv")

# 테스트 문장
sent = '내일 오전 10시에 탕수육 주문하고 싶어요.'

try:
  from utils.preprocess import Preprocess

  # 전처리 객체 생성
  p = Preprocess(userdic=user_dic_path)

  # 형태소 분석기 실행
  pos = p.pos(sent)

  # 품사 태그와 단어를 함께 출력
  ret = p.get_keywords(pos, without_tag=False)
  print('태그를 포함한 형태소 분석: ', ret)

  ret = p.get_keywords(pos, without_tag=True)
  print('태그를 제거한 형태소 분석: ', ret)
except Exception as e:
  print('error: ', e)
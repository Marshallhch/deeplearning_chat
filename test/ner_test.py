import os
import sys
from pathlib import Path
from dotenv import load_dotenv

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

load_dotenv()

# 전처리 클래스
from utils.preprocess import Preprocess

# 모델 클래스
from model.ner.ner_model import NerModel

# 전처리 객체 생성
p = Preprocess(word2index_dic=os.path.join(root_dir, 'train_tools', 'dict', 'chatbot_dict.bin'), userdic=os.path.join(root_dir, 'test', 'user_dic.tsv'))

# ner 모델 호출
ner = NerModel(model_name=os.path.join(root_dir, 'model', 'ner', 'ner_model.keras'), preprocess=p)

# 테스트 문장
query = "오늘 오전 10시에 탕수육 주문하고 싶어요"

# 개체명 예측
predicts = ner.predict(query)
print(predicts)

# 개체명 태그 예측
predict_tags = ner.predict_tags(query)
print(predict_tags)
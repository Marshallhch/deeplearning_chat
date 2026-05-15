import os
import sys
from dotenv import load_dotenv
import tensorflow as tf

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_dir)

load_dotenv()

# 커스텀 모듈
from utils.preprocess import Preprocess
from model.intent.intent_model import IntenModel

# 소프트맥스 업그레이드 버전 사용
custom_objects = {'softmax_v2': tf.nn.softmax}

# 전처리 객체 생성
p = Preprocess(word2index_dic=os.path.join(root_dir, 'train_tools', 'dict', 'chatbot_dict.bin'), userdic=os.path.join(root_dir, 'test', 'user_dic.tsv'))

# 의도 분류 모델 호출
intent = IntenModel(model_name=os.path.join(root_dir, 'model', 'intent', 'intent_model.keras'), preprocess=p, custom_objects=custom_objects)

# 테스트 질문
# query = '내일 오전 10시에 탕수육 주문하고 싶어요.'
# query = '바보 멍청이'
query = '안녕하세요. 반갑습니다.'

# 의도 클래스 예측
predict = intent.predict_class(query)

# 의도 레이블
predict_label = intent.labels[predict]

print('질문: ', query)
print('의도 클래스 예측: ', predict)
print('의도 레이블 예측: ', predict_label)
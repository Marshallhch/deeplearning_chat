import os
import sys
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd
import tensorflow as tf

from tensorflow.keras import preprocessing
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Dense, Dropout, Conv1D, GlobalMaxPooling1D, concatenate

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

load_dotenv()

# 데이터 호출
train_file = os.path.join(root_dir, 'model', 'intent', 'total_train_data.csv')
data = pd.read_csv(train_file)

# print(data[:5])

# 질문과 의도 문자열을 리스트로 변환 저장
queries = data['query'].tolist()
intents = data['intent'].tolist()

# print(queries[:5])
# print(intents[:5])

# 단어 인덱스와 사용자 정의 사전 파일 호출 및 저장
from utils.preprocess import Preprocess
p = Preprocess(word2index_dic=os.path.join(root_dir, 'train_tools', 'dict', 'chatbot_dict.bin'), userdic=os.path.join(root_dir, 'test', 'user_dic.tsv'))


# 단어 시퀀스 생성
sequences = []
for sentence in queries:
  pos = p.pos(sentence)
  keywords = p.get_keywords(pos, without_tag=True)
  seq = p.get_wordidx_sentence(keywords)
  sequences.append(seq)

# print(sequences[:5])

# 단어 인텍스 시퀀스 벡터 생성 및 입력 크기 지정(maxlen)
padded_seqs = preprocessing.sequence.pad_sequences(sequences, maxlen=15, padding='post')

# 학습용, 검증용, 테스트용 데이터셋 분리: 7:2:1
ds = tf.data.Dataset.from_tensor_slices((padded_seqs, intents))
ds = ds.shuffle(len(queries))

print(len(ds))
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import numpy as np

root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, root_dir)

load_dotenv()

# 전처리 클래스
from utils.preprocess import Preprocess

# 텐서 플로우 모듈
import tensorflow as tf
from tensorflow.keras import preprocessing
from sklearn.model_selection import train_test_split

# 학습 파일 호출
def read_ner_data(filename):
  sents = []
  with open(filename, 'r', encoding='utf-8') as f:
    lines = f.readlines() # 각 라인별 분리
    for idx, l in enumerate(lines):
      # 세미콜론으로 시작하는 줄은 문장의 헤더, 이후 문장이 $로 시작하면 새 문장 시작으로 보고 this_sents 리스트 초기화
      if l[0] == ';' and lines[idx + 1][0] == "$":
        this_sent = []
        # $로 시작하는 줄은 보통 엔티티 주석, 이전 줄이 세미콜론이면(=바로 이전 문장에 헤더가 온다면) $ 줄은 건너뜀
      elif l[0] == '$' and lines[idx - 1][0] == ';':
        continue
      # 빈 줄(문장 경계)를 만나면, 지금까지 모든 토큰들을 하나의 문장으로 보 sents에 추가
      elif l[0] == '\n':
        sents.append(this_sent)
      # 나머지 줄은 토큰 라인으로 간주해서, 공백으로 나눠 튜플 생성('1', '가락지빵', 'NNG', 'B_FOOD')
      else:
        this_sent.append(tuple(l.split()))
  return sents

# 전처리 객체 생성
p = Preprocess(word2index_dic=os.path.join(root_dir, 'train_tools', 'dict', 'chatbot_dict.bin'), userdic=os.path.join(root_dir, 'test', 'user_dic.tsv'))

# 학습용 말뭉치 데이터 호출
corpus = read_ner_data(os.path.join(root_dir, 'model', 'ner', 'ner_train.txt'))

# print(corpus[:3])

# 말뭉치 데이터에서 BIO 태그만 불러와 학습 데이터셋 생성
sentences, tags = [], []
for t in corpus:
  tagged_sentence = []
  sentence, bio_tag = [], []

  for w in t:
    tagged_sentence.append((w[1], w[3])) # , ('1', '가락지빵', 'NNP', 'B_FOOD') -> ('가락지빵', 'B_FOOD')
    sentence.append(w[1])
    bio_tag.append(w[3])
  sentences.append(sentence)
  tags.append(bio_tag)

# 데이터 확인
# print('샘플 크기: ', len(sentences))
# print('1000번째 샘플 문장: ', sentences[1000])
# print('1000번째 샘플 BIO 태그: ', tags[1000])

# 토크나이징 정의
tag_tokenizer = preprocessing.text.Tokenizer(lower=False)
tag_tokenizer.fit_on_texts(tags)

# 단어 사전 및 태그 사전 크기 정의
vocab_size = len(p.word_index) + 1
tag_size = len(tag_tokenizer.word_index) + 1

# 데이터 크기 확인
# print('단어 사전 크기: ', vocab_size)
# print('BIO 태그 사전 크기: ', tag_size)

# 학습용 단어 시퀀스 생성
x_train = [p.get_wordidx_sentence(sent) for sent in sentences]
y_train = tag_tokenizer.texts_to_sequences(tags)

index_to_ner = tag_tokenizer.index_word # 인덱스에서 개체명으로 전환
index_to_ner[0] = "PAD" # 패딩 0을 PAD 문자열로 변환

# 시퀀스 처리
max_len = 40
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=max_len, padding='post')
y_train = preprocessing.sequence.pad_sequences(y_train, maxlen=max_len, padding='post')

# 학습 데이터, 테스트 데이터 분리
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# 분리된 데이터 차원 확인
# print('학습 데이터 형태: ', x_train.shape)
# print('타겟 데이터 형태: ', y_train.shape)
# print('학습 데이터 0번째: ', x_train[0])

# 훈련(BI_LSTM) 모듈 정의
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout, Bidirectional, Input
from tensorflow.keras.optimizers import Adam

# 타겟 데이터 인코딩
y_train = tf.keras.utils.to_categorical(y_train, num_classes=tag_size)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=tag_size)

# print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# 모델 정의
model = Sequential()
model.add(Input(shape=(max_len,)))
model.add(Embedding(input_dim=vocab_size, output_dim=30))
model.add(Bidirectional(LSTM(200, return_sequences=True))) # return_sequences: 각 타임 스탭 마다의 출력을 반환
model.add(Dropout(0.5))
model.add(Dense(tag_size, activation="softmax"))

# model.summary()

# 모델 컴파일
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01), metrics=['accuracy'])

# 모델 학습 및 평가
model.fit(x_train, y_train, batch_size=20, epochs=10, validation_split=0.2)
print('평가 결과: ', model.evaluate(x_test, y_test)[1])

# 모델 저장
model.save(os.path.join(root_dir, 'model', 'ner', 'ner_model.keras'))

# F1 스코어 측정을 위하 모듈
from tensorflow.keras.models import load_model
from seqeval.metrics import f1_score, classification_report

# 모델 로드
model = load_model(os.path.join(root_dir, 'model', 'ner', 'ner_model.keras'))

# 시퀀스를 NER 태그로 변환하는 함수
def sequendce_to_tag(sequences): # 에측값을 index_to_ner를 사용해 태그 정보로 변경
  result = []
  for sequence in sequences: # 전체 시퀀스로부터 시퀀스를 하나씩 순회
    temp = []
    for pred in sequence: # 시퀀스로부터 예측 값을 하나씩 추출
      pred_index = np.argmax(pred) # [0, 0, 1, 0, 0] - 가장 큰 값 1의 인덱스 2 저장
      temp.append(index_to_ner[pred_index].replace("PAD", "0"))
    result.append(temp)
  return result

# 테스트 데이터 NER 예측
y_predicted = model.predict(x_test)
pred_tags = sequendce_to_tag(y_predicted) # 예측된 NER
test_tags = sequendce_to_tag(y_test) # 실제 NER

# 평가 결과 출력
print(classification_report(test_tags, pred_tags))
print('f1 스코어: ', f1_score(test_tags, pred_tags))
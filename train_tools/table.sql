CREATE TABLE IF NOT EXISTS chatbot_train (
  id        INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  intent    VARCHAR(100),
  ner       VARCHAR(100),
  query     VARCHAR(100),
  answer    VARCHAR(100),
  answer_image VARCHAR(2048)
);
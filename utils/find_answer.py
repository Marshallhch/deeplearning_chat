class FindAnswer:
  def __init__(self, db):
    self.db = db

  def _make_query(self, intent_name, ner_tags):
    """
    intent_name과 ner_tags를 조합하여 chatbot_train 테이블 검색용
    SQL과 바인딩 파라미터를 함께 반환한다.

    반환값: (sql: str, params: tuple)
      - sql    : ? 플레이스홀더가 포함된 SQL 문자열
      - params : ?에 순서대로 바인딩될 값들의 튜플

    사용 예)
      sql, params = self._make_query(intent, ner_tags)
      cursor.execute(sql, params)

    파라미터 바인딩을 쓰는 이유:
      - SQL Injection 방지: 외부 입력값을 SQL 문자열에 직접 이어붙이면
        악의적인 입력("'; DROP TABLE --" 등)으로 DB를 조작할 수 있다.
        ? 플레이스홀더를 쓰면 DB 드라이버가 값을 안전하게 이스케이프한다.
      - 가독성 향상: 쿼리 구조와 데이터가 분리되어 읽기 쉽다.
    """
    sql = "SELECT * FROM chatbot_train"
    params = ()  # 바인딩 값을 순서대로 쌓을 튜플

    if intent_name is not None and ner_tags is None:
      # 의도(intent)만 존재하는 경우
      # ? 는 플레이스홀더 — DB 드라이버가 params의 값으로 안전하게 대체한다
      sql += " where intent=%s"
      params += (intent_name,)

    elif intent_name is not None and ner_tags is not None:
      # 의도와 개체명(NER)이 모두 존재하는 경우
      sql += " where intent=%s"
      params += (intent_name,)

      if len(ner_tags) > 0:
        # ner 컬럼에 개체명 태그가 포함(LIKE)되는지 OR 조건으로 검색
        # 예) ner_tags = ['DT', 'LC']
        #   → SQL  : ner like %s or ner like %s
        #   → params: ('%DT%', '%LC%')
        # % 와일드카드는 파라미터 값 자체에 포함시켜 전달한다
        ner_placeholders = " or ".join("ner like %s" for _ in ner_tags)
        sql += " and (" + ner_placeholders + ")"
        params += tuple("%" + ne + "%" for ne in ner_tags)

    # 동일 조건을 만족하는 답변이 여러 건일 때 매번 다른 응답을 주기 위해 랜덤 정렬
    sql += " order by random()"

    # 최종적으로 1건만 반환 (챗봇 응답은 하나면 충분)
    sql += " limit 1"

    return sql, params

  # 답변 검색
  def search(self, intent_name, ner_tags):
    # 의도명과 개체명으로 답변 검색
    sql, params = self._make_query(intent_name, ner_tags)
    answer = self.db.select_one(sql, params)

    # 검색되는 답변이 없으면 의도명만 검색
    if answer is None:
      sql, params = self._make_query(intent_name, None)
      answer = self.db.select_one(sql, params)
    
    return (answer['answer'], answer['answer_image'])

  # NER 태그를 실제 입력된 단어로 변환
  def tag_to_word(self, ner_predicts, answer):
    for word, tag in ner_predicts:
      # q변환해야 하는 태그가 있는 경우 추가
      if tag == "B_FOOD" or tag == "B_DT" or tag == "B_TI":
        answer = answer.replace(tag, word)

    answer = answer.replace('{', '')
    answer = answer.replace('}', '')
    return answer
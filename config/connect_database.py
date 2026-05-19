import psycopg2

class Database:
  def __init__(self, host, user, password, database, port):
    self.db = None
    self.cursor = None
    try:
      self.db = psycopg2.connect(
        host=host,
        database=database,
        user=user,
        password=password,
        port=port
      )
      self.cursor = self.db.cursor()
    except Exception as e:
      print(e)

  def close(self):
    if self.db is None:
      print("데이터베이스 연결 실패")
    else:
      print("데이터베이스 연결 성공")
    self.cursor.close()
    self.db.close()

  # SQL 구문 실행
  def excute(self, sql):
    last_row_id=-1 # 반환값
    try:
      self.cursor.execute(sql)
      self.db.commit()
      last_row_id = self.cursor.lastrowid
    except Exception as e:
      print(e)
    finally:
      return last_row_id

   # SELECT 구문 실행 후 단 1개의 데이터 ROW 반환
  def select_one(self, sql, params=()):
    try:
      self.cursor.execute(sql, params)
      row = self.cursor.fetchone()
      if row is None:
        return None
      columns = [desc[0] for desc in self.cursor.description]
      return dict(zip(columns, row))
    except Exception as e:
      print(e)
      return None

  # SELECT 구문 실행 후 전체 데이터 ROW 반환
  def select_all(self, sql, params=()):
    try:
      self.cursor.execute(sql, params)
      rows = self.cursor.fetchall()
      columns = [desc[0] for desc in self.cursor.description]
      return [dict(zip(columns, row)) for row in rows]
    except Exception as e:
      print(e)
      return None
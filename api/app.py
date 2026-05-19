import socket, json
from typing import Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

app = FastAPI(
  title="Chatbot Example",
  description="Chatbot Server API",
  version="1.0.0"
)

# 소켓 접속 정보
host = '127.0.0.1'
port = 5050

# 요청 바디로 전달된 데이터 타입 정의
class QueryRequest(BaseModel):
  query: str

  class Config:
    json_schema_extra = {
      "example": {
        'query': '내일 오전 10시에 탕수육 주문 할께요'
      }
    }

# 봇 엔진 서버와 통신하는 함수
def get_answer_from_engine(bottype: str, query: str) -> Dict[str, Any]:
  try:
    # 엔진 서버 연결
    with socket.socket() as mySocket:
      mySocket.connect((host, port))

      # 질의 요청
      json_data = {
        "Query": query,
        "BotType": bottype
      }
      message = json.dumps(json_data)
      mySocket.send(message.encode())

      # 응답 수신
      data = mySocket.recv(2048).decode()
      ret_data = json.loads(data)

      return ret_data
  except Exception as e:
    raise HTTPException(status_code=500, detail=f'엔진과 통신 에러: {str(e)}')

######### 엔드포인트 작성 ##########\
@app.post("/query/{bot_type}", response_model=Dict[str, Any], description='질의 전달 및 응답 반환 봇')
async def query(
  bot_type: str,
  request: QueryRequest
):
  """
  bot_type: 봇 타입(TEST, KAKAO, LINE...) - 여기서는 TEST만 작동
  request: 질의 내용
  """
  try:
    if bot_type == "TEST":
      ret = get_answer_from_engine(bottype=bot_type, query=request.query)
      return ret
    elif bot_type == "KAKAO":
      raise HTTPException(status_code=501, detail="카카오는 지원하지 않습니다.")
    elif bot_type == "LINE":
      raise HTTPException(status_code=501, detail="라인은 지원하지 않습니다.")
    else:
      raise HTTPException(status_code=404, detail="지원하지 않는 봇 타입 입니다.")
  except HTTPException as e:
    raise HTTPException(status_code=e.status_code, detail=e.detail)
  except Exception as ex:
    print("질의 전달 오류: ", str(ex))


@app.get("/")
async def root():
  return {"message": "Hello World"}

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(
    "app:app",
    host="127.0.0.1",
    port=8000,
    reload=True,
    workers=1 # 프로세스 수
  )
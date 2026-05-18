# 싱글 스레드는 한 번에 하나의 작업만 실행한다.
# 따라서 여러 클라이언트의 응답 처리를 하기가 어려워 진다.
# 이런 경우 멀티 스레드 방식으로 엔진 서버를 만든다.
# 멀티 스레드가 가능하도록 구현하는 모듈을 TCP 소켓이라 한다.
import socket

class BotServer:
  """
  srv_port: 소켓 서버 포트
  listen_num: 최대 연결 수
  """
  def __init__(self, srv_port, listen_num):
    self.port = srv_port
    self.listen = listen_num
    self.sock = None

  # sock 생성 함수
  # 파이썬에서 지원하는 저수준 네트워킹 인터페이스 API를 사용
  # TCP/IP 소켓을 생성한 뒤 지정한 서버 포트로 지정한 연결 수를 생성
  def create_sock(self):
    """
    socket.AF_INET: IPv4 주소 체계
    socket.SOCK_STREAM: 연결 지향 소켓
    """
    self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    self.sock.bind(('0.0.0.0', int(self.port)))
    self.sock.listen(int(self.listen))

  # 크라이언트 대기
	# 서버 소켓은 설정한 주소와 통신에 바인드되어 클라이언트 연결을 주시하고 있어야 한다.
	# 클라이언트가 연결을 요청하는 즉시 accept() 함수를 호출하여 클라이언트와 통신할 수 있도록 한다.
	# 이때 반환값은 (conn, address) 형식의 튜플이다
	# conn: 클라이언트와 통신할 수 있는 소켓
	# address: 클라이언트의 주소
  def ready_for_client(self):
    return self.sock.accept()
  
  # 소켓 최종 반환
  def get_sock(self):
    return self.sock
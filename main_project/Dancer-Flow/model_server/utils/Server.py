import socket

class Server:
    def __init__(self, Ip, srv_port, listen_num):
        self.port = srv_port
        self.listen = listen_num
        self.mySock = None
        self.IP = Ip

    # sock 생성
    def create_sock(self):
        self.mySock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.mySock.bind((self.IP, int(self.port)))
        self.mySock.listen(int(self.listen))
        return self.mySock

    # client 대기
    def ready_for_client(self):
        return self.mySock.accept()

    # sock 반환
    def get_sock(self):
        return self.mySock

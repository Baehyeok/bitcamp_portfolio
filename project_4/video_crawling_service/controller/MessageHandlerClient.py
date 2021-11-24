import json

class MessageHandlerClient:
    key1 = 'menu'
    key2 = 'data'

    def __init__(self) -> None:
        pass

    def create_msg(self, str : str, num : int = 1, type='json'):
        if type=='json':
            return json.dumps({ self.key1 : num, self.key2 : str})
        else:
            return{ self.key1 : num, self.key2 : str}

    def send(self, sock, send_data : str = ""):
        if send_data == "":
                send_data = input('입력>>')
        sock.send(bytes(send_data.encode()))

    def recv(self, sock):
        while True:
            recv_data = sock.recv(1024).decode()
            print(recv_data)
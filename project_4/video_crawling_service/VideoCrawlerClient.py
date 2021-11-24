import socket
import threading
from view.ClientView import ClientView
from controller.MessageHandlerClient import MessageHandlerClient
import json


class VideoCrawlerClientMain:
    IP = 'localhost'
    PORT = 9999

    def __init__(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.handler = MessageHandlerClient()
        self.view = ClientView()

    def main(self):
        self.connect()
        while True:
            self.view.show_crawl_menu()
            num = self.view.select_menu_num()
            result = self.handle_menu(num)
            if result < 0:
                break

    def connect(self):
        self.socket.connect((self.IP, self.PORT))
        print('Connected to ', self.IP, self.PORT)

        recv_thread = threading.Thread(target=self.handler.recv, args=(self.socket,))
        recv_thread.start()

    def handle_menu(self, num):
        if num == self.view.OPTION.CRAWLING:
            options = self.view.print_crawl_option()
            msg = self.handler.create_msg(json.dumps(options), num)
            self.handler.send(self.socket, msg)
            
        elif num == self.view.OPTION.SHOW_LIST:
            pass
        elif num == self.view.OPTION.DOWNLOAD:
            pass
        elif num == self.view.OPTION.SEARCH:
            pass
        elif num == self.view.OPTION.EXIT:
            print('종료')
            return -1
        else:
            print('다시 입력해주세요')
        
        return 0
    

if __name__ == '__main__':
    client = VideoCrawlerClientMain()
    client.main()




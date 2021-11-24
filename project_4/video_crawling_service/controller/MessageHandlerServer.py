import json
from src.Oracle_DAO import OracleDao
from src.Sqlite_DAO import SqliteDao
from controller.DBManager import DBManager
from controller.VideoCrawlerManager import VideoCrawlerManager
from src.Option import Option


class MessageHandlerServer():

    def __init__(self) -> None:
        self.crawl_manager = VideoCrawlerManager()
        self.sqlite_manager = DBManager(SqliteDao, {'filename' : 'data/youtube.db'})
        self.oracle_manager = DBManager(OracleDao, {'ips': 'localhost', 'id' : 'bitai', 'ports' : '1521', 'pws' : 'bitai'})
        
    def handle_data(self, data, conn, count, send_queue):
        num = data['menu']
        if num == Option.CRAWLING:
            result = self.crawl_manager.start_crawling(data['data'])
            print(result)
            send_queue.put([result, conn, count])
            self.sqlite_manager.insert(result)
            self.oracle_manager.insert(result)
        elif num == Option.SEARCH:
            pass
        elif num == Option.SHOW_LIST:
            pass
        elif num == Option.DOWNLOAD:
            pass
        elif num == Option.EXIT:
            pass
        else:
            print('undefined menu')

    def send(self, group, send_queue):
        while True:
            try:
                recv = send_queue.get() 
                if recv == 'Group Changed': 
                    print('Group Changed') 
                    break

                for conn in group:
                    msg = 'Client' + str(recv[2]) + ' >> ' + str(recv[0])
                    conn.send(bytes(msg.encode())) 
            except: 
                pass

    def recv(self, conn, count, send_queue):
        while True:
            msg = conn.recv(1024).decode()
            data = json.loads(msg)
            data['data'] = json.loads(data['data'])
            print('recieved : ', data)
            send_queue.put([data, conn, count])
            self.handle_data(data, conn, count, send_queue)
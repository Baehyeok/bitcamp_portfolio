from channels.generic.websocket import WebsocketConsumer
import os
import threading
import json
from urllib.parse import urlparse
import base64
import socket
import json
import time
import asyncio
from queue import Queue
import numpy
from server.settings import PROJECT_DIR

class PlayConsumer(WebsocketConsumer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.CHUNK_FILE_PATH = 'media/chunks'
        self.MODULE_DIR = os.path.dirname(__file__)
        self.APP_PATH = os.getcwd()
        self.stamp = 0
        self.PID = None
        self.model_thread : threading.Thread = None
        self.model_socket = None
        self.total_score = 0
        self.parts_score = {
            'face_body' : 0,
            'left_arm' : 0,
            'right_arm' : 0,
            'left_leg' : 0,
            'right_leg' : 0,
        }
        self.high_chunk_score = 0
        self.high_score_chunk_path = ''
        
        self.chunk_path_queue = Queue()

    def connect(self):    # 클라이언트 연결
        self.accept()
        print(self.scope)
        self.PID = self.scope['url_route']['kwargs']['play_id']    
        self.model_socket = self.create_model_socket()
    
    def create_model_socket(self):     # Model 서버에 연결하기 위한 Client Socket 생성
        host = "127.0.0.1"  # 모델 엔진 서버 IP 주소
        port = 5050  # 모델 엔진 서버 통신 포트
        model_socket = socket.socket()
        model_socket.connect((host, port))
        print('model server connected')
        return model_socket
            
    def model_task(self, model_socket, chunk_path, stamp):  # 코루틴으로 분기하여 처리될 기능 
        
        # Model Server Config Message Send 
        self.sendToModel('chunk', {
            "pid" : self.PID,
            "path" : self.chunk_path_queue.get(),
        })
        chunk_score_sum = 0
        
        # Scoring
        while True:
            try:
                data = self.model_socket.recv(2048)
                ret_data = json.loads(data.decode())
            except Exception as e:
                print('model_task() : socket error', e)
                self.model_socket.close()
                self.model_socket = self.create_model_socket()
                break

            if ret_data["ING"] == "finish":
                # cv2.destroyAllWindows()
                break

            print("답변 : ")
            print(ret_data['total_score'])
            
            chunk_score = int(ret_data['total_score'])
            chunk_score_sum += chunk_score
            self.total_score += chunk_score
            # print(self.total_score)

            length = self.recvall(self.model_socket, 64)
            length1 = length.decode('utf-8')
            stringData = self.recvall(self.model_socket, int(length1))
            data = numpy.frombuffer(base64.b64decode(stringData), numpy.uint8)
            # decimg = cv2.imdecode(data,1)

            # print(base64.b64encode(data).decode('ascii'))
            # print(data)
            self.updateScore(chunk_score, ret_data['parts_score'], base64.b64encode(data).decode('ascii'))
            # self.sendSkeletonImage(base64.b64encode(data).decode('ascii'), data)
            
            # text = "Score : {}".format(str(ret_data['total_score']))
            # org=(50,100)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(decimg, text,org, font,1,(255,0,0),2 )
            
            # cv2.imshow("image", decimg)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
                    
        #     cv2.destroyAllWindows()
        # cv2.destroyAllWindows()

        # Update Play Preview Data
        print('sum : ', chunk_score_sum)
        if(stamp > 2 and chunk_score_sum > self.high_chunk_score):
            self.high_chunk_score = chunk_score_sum
            self.high_score_chunk_path = chunk_path
            self.updatePreviewPath(self.high_score_chunk_path)
            print('preview path : ', self.high_score_chunk_path)


    def recvall(self, sock, count):
        buf = b''
        while count:
            newbuf = sock.recv(count)
            if not newbuf: return None
            buf += newbuf
            count -= len(newbuf)
        return buf
        

    def receive(self, text_data=None, bytes_data=None, **kwargs):   # WebSocket Client로 부터 데이터를 받을때 실행된다
        if bytes_data != None:  # Chunk 데이터를 저장한다.
            self.stamp += 1
            chunk_path = os.path.join(self.CHUNK_FILE_PATH, f"{self.PID}_{self.stamp}.mp4".replace('/', '', 1)).replace('\\', '/')
            data_path = os.path.join(self.APP_PATH, f"../{chunk_path}").replace('\\', '/')

            with open(data_path, 'wb') as f:
                f.write(bytes_data)
                self.chunk_path_queue.put(data_path)
                try:
                    asyncio.create_task(self.model_task(self.model_socket, chunk_path, self.stamp))
                except Exception as e:
                    print('model socket error', e)
                return

        if text_data != None:   # Text 메시지를 처리한다.     
            self.handle_text_data(json.loads(text_data))
            
        
    def handle_text_data(self, data):
            if data['type'] =='close':
                self.disconnect(data['pid'])
            elif data['type'] == 'check':
                message = data['message']
                print(message)
                self.sendMessage(message=message)
            elif data['type'] == 'sync':
                user_src : str = data['user_video_src']
                play_src : str = data['play_video_src']
                user_src = os.path.join(PROJECT_DIR, user_src)
                play_src = os.path.join(PROJECT_DIR, play_src)
                print(user_src)
                print(play_src)
                self.sendToModel('sync', {
                    'pid' : self.PID,
                    'user_path' : user_src,
                    'play_path' : play_src,
                })
                try:
                    res = self.model_socket.recv(2048)
                    ret_data = json.loads(res.decode())

                    self.sendSync(ret_data)
                except Exception as e:
                    print('model_task() : socket error', e)
                    self.model_socket.close()
                    self.model_socket = self.create_model_socket()

                    

    def sendToModel(self, type, data):
        user_data = {
            'type' : type,
            **data,
        }
        message = json.dumps(user_data)
        self.model_socket.send(message.encode())

    def sendSync(self, data):
        self.send(text_data=json.dumps({
            'type' : 'sync',
            **data,
        }))

    def sendSkeletonImage(self, str_data, byte_data):
        self.send(text_data=json.dumps({
            'type' : 'skeleton',
            'image' : str_data,
        }), bytes_data=byte_data)

    def sendMessage(self, message):
        self.send(text_data=json.dumps({
            'type' : 'check',
            'message' : message,
            'result' : '200',
        }))

    def updateScore(self, score, parts_score, image):  # 업데이트 점수를 WebSocket Client로 전송
        self.send(text_data=json.dumps({
            'type' : 'update_score',
            'score' : score,
            'image' : image,
            'parts_score' : parts_score, 
            'result' : 200, 
        }))

    def updatePreviewPath(self, chunk_path):
        self.send(text_data=json.dumps({
            'type' : 'update_preview_path',
            'chunk_path' : chunk_path,
            'result' : 200, 
        }))         

    def disconnect(self, code):
        self.stamp = 0
        return super().disconnect(code)

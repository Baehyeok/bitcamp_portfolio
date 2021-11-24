from copy import Error
import cv2
import numpy
import base64
import socket
import json
import os
import asyncio
import time


# 챗봇 엔진 서버 접속 정보
host = "127.0.0.1"  # 챗봇 엔진 서버 IP 주소
port = 5050  # 챗봇 엔진 서버 통신 포트

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def task(mySocket):

    # 챗봇 엔진 답변 받기
    while True:
        try:
            # time.sleep(1)
            data = mySocket.recv(2048)
            ret_data = json.loads(data.decode('utf-8'))
            
        except Exception as e:
            print('error:', e)
            break
                    
        if ret_data["ING"] == "finish":
            cv2.destroyAllWindows()
            break
        else:
            print("답변 : ")
            print(ret_data['total_score'])
        length = recvall(mySocket, 64)
        length1 = length.decode('utf-8')
        stringData = recvall(mySocket, int(length1))
        data = numpy.frombuffer(base64.b64decode(stringData), numpy.uint8)
        decimg = cv2.imdecode(data,1)


        text = "Score : {}".format(str(ret_data['total_score']))
        org=(50,100)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(decimg, text,org, font,1,(255,0,0),2 )
        
        
        
        cv2.imshow("image", decimg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
                
    cv2.destroyAllWindows()

    # print(ret_data)
    # print(type(ret_data))
    # print("\n")

    # 챗봇 엔진 서버 연결 소켓 닫기


if __name__ == '__main__':

    while True:


        # 클라이언트 프로그램 시작
        print("질문 : ")
        query = input()  # 질문 입력
        if(query == "exit"):
            exit(0)
        print("-" * 40)
        # 챗봇 엔진 서버 연결
        mySocket = socket.socket()
        mySocket.connect((host, port))

        # 모델 엔진 질의 요청
        json_data = {
                "pid" : "qqq",
                # "target" : os.path.join(os.getcwd(), 'model_server/module/sample_data/dynamite.mp4') ,
                # "path" : os.path.join(os.getcwd(), 'model_server/module/sample_data/samplieclip.mp4') ,
                "chunk_path" : 'C:\project\Dancer-Flow-model\Dancer-Flow\media\chunks\\' + query ,
        }
        message = json.dumps(json_data)
        mySocket.send(message.encode())
        try:
            asyncio.create_task(task(mySocket))
        except Exception as e:
            print('async error' ,e)
            continue


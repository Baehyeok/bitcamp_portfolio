import pandas as pd
from bs4 import BeautifulSoup
import urllib.request as req

result = []

## url 생성 파라미터
html = 'https://openapi.its.go.kr:9443/cctvInfo?'       #기본 url 주소
param = 'apiKey=' + '6ca3cab8165b494db0ad7be75b9b67dd'  #apikey(공개키)
param += '&type=' + 'ex' + 'its'                        #도로 유형(ex: 고속도로 / its: 국도)
param += '&cctvType='+ '2'                              #CCTV 유형(1: 실시간 스트리밍(HLS) / 2: 동영상 파일 / 3: 정지 영상)
param += '&minX=' +    '127.0955868'                    #최소 경도 영역
param += '&maxX=' +   '127.2125905'                     #최대 경도 영역
param += '&minY=' +  '36.75464338'                      #최소 위도 영역
param += '&maxY=' +  '36.86849361'                      #최대 위도 영역
param += '&getType=' + 'xml'                            #출력 결과 형식(xml, json / 기본: xml)
url = html + param
print(url)

# webclawing
res = req.urlopen(url)
soup = BeautifulSoup(res, 'html.parser')
title = soup.find_all('data')
for a in title:
    cctvurl = a.find('cctvurl').string
    coordy = a.find('coordy').string
    coordx = a.find('coordx').string
    cctvname = a.find('cctvname').string
    result.append([cctvurl]+[coordy]+[coordx]+[cctvname])

# DataFrame으로 저장하기
df = pd.DataFrame(result)
df.to_csv('data/cctv_csv/cctv_url_all.csv')



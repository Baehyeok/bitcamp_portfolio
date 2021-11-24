from dateutil.relativedelta import relativedelta
from bs4 import BeautifulSoup
from itertools import count
import urllib.request
import pandas as pd
import datetime
import ssl

searchname_list = ['소주', '전통주', '위스키', '럼', '보드카', '막걸리', '지역소주', '고량주']
numbers = int(12)
result = []
dateformat = "%Y.%m.%d"
dateformatnso = "%Y%m%d"
today = datetime.datetime.now()
context = ssl._create_unverified_context()
def get_request_url(url,enc='utf-8'):
    response = urllib.request.urlopen(url)
    request = urllib.request.Request(url, headers={'USer-Agent':'Mozilla/5.0'})
    try:
        response = urllib.request.urlopen(request)
        if response.getcode() == 200:
            print("[%s] Url Request Success" % datetime.datetime.now() + tmp + searchname+ "(" +str(page+1)+")")
            return response.read().decode(enc)
    except Exception as e:
        print(e)
        print("[%s] Error for URL : %s" %(datetime.datetime.now(), url))
        return None
for searchname in searchname_list:
    for num in count():
        tm = datetime.datetime(2021,1,1) - relativedelta(months=num)
        tmp = tm.strftime(dateformat)
        tmpnso = tm.strftime(dateformatnso)
        beday= datetime.datetime(2021,1,1) - relativedelta(months=(num+1))
        bmp = beday.strftime(dateformat)
        bmpnso = beday.strftime(dateformatnso)
        if num > int(numbers)-int(1):
            break
        for page in count():
            endpoint = 'https://search.naver.com/search.naver'
            params = '?where=news&sm=tab_jum'                       
            params += '&query=' + urllib.parse.quote(searchname)   
            params += '&sort=1'                                   
            params += '&ds=' + tmp                                  
            params += '&de=' + bmp                                  
            params += '&start=' + str(page)+str(1)                 
            params += '&nso=so%3Add%2Cp%3Afrom' + tmpnso + 'to' + bmpnso
            url = endpoint+params
            if page > 399:
                print("뉴스의 개수가 4000개가 넘었습니다.")
                break
            try: 
                respone = get_request_url(url)
                soupData = BeautifulSoup(respone, 'html.parser')
                table = soupData.find('ul',{'class' : 'list_news'})
                uls = table.find_all('div',{'class':"news_area"})
                for store in uls:
                    title = store.find('a', {'class' : 'news_tit'}).text
                    yyyymm = store.find('span',{'class':"info"}).text
                    if yyyymm.endswith('P') == True:
                        break
                    elif yyyymm.endswith(str('단')) == True:
                        break
                    content = store.find('a',{'class':"api_txt_lines dsc_txt_wrap"}).text
                    try:
                        namees = store.find('a',{'class':'info press'}).text
                    except AttributeError as e:
                        break
                    result.append([yyyymm]+[namees]+[title]+[content])
            except AttributeError as e:
                break
    df =pd.DataFrame(result)
    df.to_csv("./data/네이버%s뉴스_2020년도.csv" %(searchname))
    print("%s파일저장완료"%(searchname))
    result = []
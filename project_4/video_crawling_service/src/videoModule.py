from bs4 import BeautifulSoup as bs 
from selenium import webdriver as wb 
from pytube import YouTube
from moviepy.editor import *
from selenium.webdriver.common.keys import Keys
import pandas as pd
import re
import time
import os 


class VideoModul:
    def VideoCrawler(self, options):
        "main에서 옵션으로 searchname을 받으면 name으로 링크, 이름, 시간, 조회수 등을 DB에 저장한다."
        data = []
        driver = wb.Chrome('driver/chromedriver.exe') 
        search_keywords = "+".join(options['keywords'])
        url = ('https://www.youtube.com/results?search_query=%s&sp=EgQQASAB'%(search_keywords))
        driver.get(url)
        body = driver.find_element_by_tag_name('body')
        body.send_keys(Keys.PAGE_DOWN)

        # 스크롤 0.5초마다 한번씩 총 100번 내리기
        for i in range(1,101):
            body.send_keys(Keys.PAGE_DOWN)
            time.sleep(0.5)

        soup = bs(driver.page_source, 'lxml')
        title =soup.select('a#video-title')
        times = soup.select('span#text')
        video_url = soup.select('a#video-title')

        view = soup.select('div#metadata-line > span:nth-of-type(1)')

        title_list = []
        view_list = []
        times_list = []
        url_list=[]
        origin_list = []

        for i in range(len(view)):
            tt = times[i].text.split(':')
            if len(tt) == 2 and int(tt[0]) >= options['time'] or len(tt) == 3:
                continue
            title_list.append(re.sub("([^(가-힣)(ㄱ-ㅎ)(ㅏ-ㅣ)(a-z)(A-Z)(\s)])+","", title[i].text.strip()))
            view_list.append(view[i].text.strip())
            times_list.append(times[i].text.strip())
            origin_list.append("Youtube")
            url_list.append('{}{}'.format('https://www.youtube.com',video_url[i].get('href')))

        info = {'link':url_list, 'time':times_list, 'title':title_list, 'view':view_list, 'origin':origin_list, 'search_keywords' : search_keywords}
        data = pd.DataFrame(info)
        data.to_csv("data/%s.csv"%(search_keywords))

        return data
        
    def VideoDownload(self, searchname, storage):
        "searchname 입력받으면 DB 에서 name을 검색해서 링크를 받아와 다운로드 한다."
        data = pd.read_csv('%s.csv'%(searchname))
        for urls, name in zip(data['link'], data['title']):
            url = urls
            yt = YouTube(url)
            vids = yt.streams.all()
            for i in range(len(vids)):
                print(i,'. ',vids[i])
            yt.streams.filter(adaptive=True,
                                file_extension='mp4',
                                only_video=True).order_by('resolution').desc().first().download(storage,'%s.mp4'%(name))
            yt.streams.filter(adaptive=True,
                                file_extension='mp4',
                                only_audio=True).order_by('abr').desc().first().download(storage,'%s.mp3'%(name))

            # 영상 합치기 작업(영상 mp4+ 음성 mp3)
            # videoclip = VideoFileClip("C:/JGBH/Project/final_project/youtube/test.mp4")
            # audioclip = AudioFileClip("C:/JGBH/Project/final_project/youtube/test.mp3")
            # videoclip.audio = audioclip
            # videoclip.write_videofile("test2.mp4")

if __name__ == '__main__':
    storage_path = os.getcwd() + '\\data'
    searchname = input("검색할 단어를 입력하시오 :")
    VideoModul.VideoCrawler(searchname)
    # VideoModul.VideoDownload(searchname, storage_path)

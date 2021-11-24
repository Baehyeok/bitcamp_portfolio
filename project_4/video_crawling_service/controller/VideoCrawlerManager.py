import json
import sys
from pandas.core.frame import DataFrame
from .Manager import Manager
from src.videoModule import VideoModul


class VideoCrawlerManager(Manager):
    def __init__(self):
        self.crawler = VideoModul()

    def start_crawling(self, option: str):
        result_code = self.RESULT_CODE['FAILED']
        data = self.crawler.VideoCrawler(option)
        if type(data) == DataFrame:
            result_code = self.RESULT_CODE['SUCCESS']
        else:
            result_code = self.RESULT_CODE['Error']
        self.result_handler(result_code, 'start_crawling')
        return data 

    def start_download(self, option: str):
        option_dict = json.load(option)
        result = self.crawler.download(option_dict)
        self.result_handler(result, 'start_download')

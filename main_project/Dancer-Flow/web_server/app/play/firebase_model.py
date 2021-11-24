

from datetime import datetime

from firebase_admin import firestore
import os
import pytz
KST = pytz.timezone('Asia/Seoul')


class Play(object):
    def __init__(self, option_mode='practice', option_upload='upload', songs=[], user='', upload_path='', \
            play_path='', date='', status='play', total_score=0, parts_score={}, \
            title='', content='', tags=[], faves=0, views=0, isShared=False):
        self.option_mode = option_mode
        self.option_upload = option_upload
        self.upload_path = os.path.join('/', upload_path) 
        self.play_path = os.path.join('/', play_path)
        self.songs = songs
        self.user = user
        self.date = datetime.now(KST) if date == ''  else date 
        self.status = status
        self.parts_score = parts_score
        self.total_score = total_score
        self.title = title
        self.content = content
        self.tags = tags
        self.faves = faves
        self.views = views
        self.isShared = isShared

    @staticmethod
    def from_dict(source):
        pass

    def to_dict(self):
        return self.__dict__

    def __repr__(self) -> str:
        return super().__repr__()
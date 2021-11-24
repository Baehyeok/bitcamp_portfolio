
from server.settings import DB
from datetime import datetime
import pytz
KST = pytz.timezone('Asia/Seoul')


class User(object):
    CLT_NAME = 'User'
    NAME_FIELD = 'username'
    EMAIL_FIELD = 'email'
    FAVES_FIELD = 'faves_list'

    def __init__(self, username, password, email='', join_date='', faves_list=[], views_list=[], image='../static/images/user.png' ):
        self.username = username
        self.password= password
        self.email = email
        self.join_date = datetime.now(KST) if join_date == '' else join_date
        self.faves_list = faves_list
        self.views_list = views_list
        self.image_path = image

    @classmethod
    def isExistsID(cls, user_id):
        doc_ref = DB.collection(cls.CLT_NAME).document(user_id)
        if doc_ref.get().exists:
            return doc_ref.get()
        else:
            return False

    @classmethod
    def isExistsUserName(cls, username):
        doc = DB.collection(cls.CLT_NAME).where(cls.NAME_FIELD, u'==', username).get()
        if len(doc) == 1:
            return doc[0]
        else:
            return None


    @classmethod
    def isExistsEmail(cls, email):
        doc = DB.collection(cls.CLT_NAME).where(cls.EMAIL_FIELD, u'==', email).get()
        if len(doc) == 1:
            return doc[0]
        else:
            return None

    @classmethod
    def Login(cls, username, password):
        users = DB.collection(cls.CLT_NAME).where(cls.NAME_FIELD, u'==', username).stream()
        print(users)
        for user in users:
            print(user)
        return users

    def register(self):
        DB.collection(self.CLT_NAME).add(self.to_dict())


    @staticmethod
    def from_dict(source):
        pass

    def to_dict(self):
        return self.__dict__

    def __repr__(self) -> str:
        return super().__repr__()
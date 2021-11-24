from sqlite3.dbapi2 import DateFromTicks
from .Manager import Manager

class DBManager(Manager):
    def __init__(self, DAO, options):
        self.dao = DAO(options)
        
        
    # def connect(self, user, db, table):
    #     result_code = self.oracle_dao.connect(user, db, table)
    #     self.result_handler(result_code, 'connect')

    # def update(self, sql: str):
    #     result_code = self.oracle_dao.update(sql)
    #     self.result_handler(result_code, 'update')
 
    def select_table(self):
        result_code = self.dao.searchVideo()
        self.result_handler(result_code, 'select_table')

    def select_content(self, tableName: str):
        result_code = self.dao.searchVideo(tableName)
        self.result_handler(result_code, 'select_content')

    def insert(self, data):
        # "{'link', 'time', 'title', 'view','origin'}"
        self.dao.dfToTable(data)

    def delete(self, tableName: str):
        result_code = self.dao.insertVideo(tableName)
        self.result_handler(result_code, 'delete')
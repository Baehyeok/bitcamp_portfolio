import sqlite3

from pandas.core.frame import DataFrame

class SqliteDao:
    def __init__(self, options):
        "DB 접속 파일 인자 받기"
        self.options = options
        self.conn = sqlite3.connect(self.options['filename'], check_same_thread=False)
        self.cursor = self.conn.cursor()

    def __del__(self):
        "cursor, conn 종료"
        self.cursor.close()
        self.conn.close()

    def CreateTable(self, tableName):
        "테이블 생성 함수" 
        drop_sql = """
            DROP TABLE IF exists video_%s;
        """%(tableName)
        self.cursor.execute(drop_sql)
        create_sql = ("""
            CREATE TABLE video_%s(
                id INTEGER NOT NULL,
                link TEXT NOT NULL,
                time TEXT,
                title TEXT,
                view TEXT,
                origin TEXT,
                PRIMARY KEY(id AUTOINCREMENT)
            );
        """%(tableName))
        self.cursor.execute(create_sql)
    
    def insertVideo(self, tableName, link, time, title, view, origin):
        insert_sql = """
            INSERT INTO video_%s (link, time, title, view, origin) values('%s', '%s', '%s', '%s', '%s')
        """%(tableName, link, time, title, view, origin)
        self.cursor.execute(insert_sql)
        self.conn.commit()

    def dfToTable(self, df : DataFrame):
        table_name = '_'.join(['video'] + df.search_keywords[0].split('+'))
        df.to_sql(table_name, self.conn, if_exists='replace')

    def searchVideo(self):
        "테이블 명으로 파일을 나누고 검색한다"
        search_sql = """
            SELECT name 
            FROM sqlite_master 
            WHERE type = 'table' 
            AND name NOT LIKE 'sqlite_%'
        """
        self.cursor.execute(search_sql)
        print(self.cursor.fetchall())
        
    def deleteVideo(self, tableName):
        "테이블을 지운다"
        delete_sql = """
            DROP TABLE video_%s
        """%(tableName)
        self.cursor.execute(delete_sql)
        self.conn.commit()

    def searchcontent(self, tableName):
        sql = """
            select * from video_%s
        """%(tableName)
        [*rows] = self.cursor.execute(sql)
        return rows 
        
        
if __name__ == '__main__':
    g_sqliteDao = SqliteDao(filename='youtube.db')
    g_sqliteDao.CreateTable(tableName="올림픽")

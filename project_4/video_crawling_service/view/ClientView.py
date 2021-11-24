from src.Option import Option

class ClientView:
    className = "ClientView"
    OPTION = Option()    

    def __init__(self) -> None:
        self.keywords = []
        self.time = 10
        pass

    
    @classmethod
    def show_crawl_menu(cls):
        print('\n\n')
        print('1. Video Crawling')
        print('2. Video content')
        print('3. Download Video ')
        print('4. Search Video')
        print('5. Exit')

    @classmethod 
    def select_menu_num(self):
        return int(input("select Menu >> "))

    def print_crawl_option(self):
        self.keywords = input('원하는 검색어 입력(여러개일경우 띄어쓰기) >>').split()
        self.time = input('원하는 영상 최대 길이 입력(분) >>')
        return self.get_option()

    def get_option(self):
        return {'keywords' : self.keywords, 'time' : int(self.time)}        




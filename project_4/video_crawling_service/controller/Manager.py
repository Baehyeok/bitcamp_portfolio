
class Manager:
    RESULT_CODE = {"SUCCESS": 200, 'FAILED': 500, 'ERROR': 400}

    def __init__(self):
        pass

    def result_handler(self, result_code, func_str):
        if result_code == self.RESULT_CODE['SUCCESS']:
            print('[SUCCESS!] %s ' % func_str)
        elif result_code == self.RESULT_CODE['FAILED']:
            print('[FAILED!] %s ' % func_str)
        elif result_code == self.RESULT_CODE['ERROR']:
            print('[ERROR!] %s ' % func_str)

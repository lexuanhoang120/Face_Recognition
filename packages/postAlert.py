import datetime
import logging
import threading
import time
# from packages.alertCheck import alertCheck
import traceback


from packages.alertCheck import AlertCheck


        


def post_alert(receiver_pipe):
    global codes
    codes = set()

    def receiverCode(receiver):
        global codes
        while True:
            code = receiver.recv()
            codes.add(code)

    def alert_all():
        global codes
        alert = AlertCheck()
        while True:
            if len(codes) == 0:
                continue
            for code in codes.copy():
                alert.alert(code)

    thread = threading.Thread(target=receiverCode, args=(receiver_pipe,))
    thread.start()
    thread2 = threading.Thread(target=alert_all, args=())
    thread2.start()

    while True:
        try:
            if len(codes) == 0:
                continue
            for code in codes.copy():
                # code : ma nhan vien
                # date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # date = "2022-08-18 16:35:55"
                post_to_of1(code)
                codes.remove(code)


        except:
            logging.error(f'Process postAlert: {traceback.format_exc()}')


def post_to_of1(code):
    try:
        if code[:6] == 'VTCODE':
            
            logging.info(f"POST,1,{code}")
    except:
        logging.info(f"Not post 1office:{code}")



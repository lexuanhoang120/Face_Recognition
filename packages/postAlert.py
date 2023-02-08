import datetime
import logging
import threading
import time
# from packages.alertCheck import alertCheck
import traceback

import requests

from packages.alertCheck import AlertCheck
import pymssql

import pyodbc
def connect_database_vtcode():
    try:
        """
            insert information to database checkin checkout and update checkout
            """
        driver = 'ODBC Driver 17 for SQL Server'
        server = 'rdp.icool.com.vn,1434'
        database = 'cham_cong_db'
        username = 'data'
        password = 'D@ta!@#$'
        database = pyodbc.connect(f"""
                DRIVER={driver};
                SERVER={server};
                DATABASE={database};
                UID={username};
                PWD={password};
            """)
        return database
    except:
        server = 'rdp.icool.com.vn:1434'
        database = 'cham_cong_db'
        username = 'data'
        password = 'D@ta!@#$'
        database = pymssql.connect(server=server,
                                    user=username, 
                                    password=password, 
                                    database=database)
        return database 

# function insert info to database
def insert_information_vtcode(code):
    try:
        database = connect_database_vtcode()
        def has_existed(cursor, code):
            cursor.execute(f"""select ID from VaoRa_Camera
                            where MaNV= '{code}' and
                                cast(Ngay as date) = cast(GETDATE() as date)
                                order by ThoiGian desc
                            """)
            result = cursor.fetchall()
            try:
                if len(result) == 0:
                    return None, 0
                else:
                    return result, len(result)
            except:
                logging.error(f"has_existed_VaoRa_Camera:{traceback.format_exc()}")
                return None, 0

        cursor = database.cursor()
        uid, number_ = has_existed(cursor, code)
        
        if number_ == 0:
            query = f"""
                insert into 
                    VaoRa_Camera(MaNV,ThoiGian,Kieu,May,Ngay)
                values
                ('{code}',FORMAT(GETDATE(),'yyyy-MM-dd HH:mm:ss'),0,1999,FORMAT(GETDATE(),'yyyy-MM-dd'))
                                """
        elif number_ == 1 :
            query = f"""
                insert into 
                    VaoRa_Camera(MaNV,ThoiGian,Kieu,May,Ngay)
                values
                ('{code}',FORMAT(GETDATE(),'yyyy-MM-dd HH:mm:ss'),1,1999,FORMAT(GETDATE(),'yyyy-MM-dd'))
  
                                """
        else:
            query = f"""update VaoRa_Camera
                        set ThoiGian = FORMAT(GETDATE(),'yyyy-MM-dd HH:mm:ss')
                        where ID={uid[0][0]};
                                """    
        cursor.execute(query)
        database.commit()
        del query
        return 0
    except:
        logging.error(f"Code: {code} not insert database chamcongcamera")
        logging.error(f",Insert_information: {traceback.format_exc()}")
        



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
            code = int(code[6:])
            insert_information_vtcode(code)
            url = "https://space.1office.vn/timekeep/attendance/service"
            data = '''[{ ''' + f'''
                            "code": "{code}", 
                            "time":"{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}",
                            "machine_code":"faceCam", 
                            "machine_ip":"192.168.0.201" ''' + '''}]'''
            # headers = { 'Cookie': 'PHPSESSID=6can653vsteb4s83nqar5c458n' }
            result = requests.request(method="POST",
                                    url=url,
                                    headers={},
                                    data={'key': 'space', 'data': data},
                                    files=[]).text
            logging.info(f"POST,1,{code},{result}")
    except:
        logging.info(f"Not post 1office:{code}")



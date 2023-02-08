"""
Author : Xuan Hoang
Data : 25/11/2022
Function : find matching face in faces


"""
# define a log file for tracking the processes
import datetime
import logging
import os
import sqlite3
import traceback

import cv2
import pyodbc
import pymssql 

# A process is to receive the code and post to 1office in order to checkin-out

# a function to save image when staff is checkin
def connect_database():
    try:
        """
            insert information to database checkin checkout and update checkout
            """
        driver = 'ODBC Driver 17 for SQL Server'
        server = 'rdp.icool.com.vn,1434'
        database = 'vtcode_attendance'
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
        database = 'vtcode_attendance'
        username = 'data'
        password = 'D@ta!@#$'
        database = pymssql.connect(server=server,
                                    user=username, 
                                    password=password, 
                                    database=database)
        return database


def save_image(image, code):
    """
    save image
    input: img:array, name:string
    output: save image from input with name file using name input
    """
    try:
        path_img = "images/" + str(datetime.date.today())
        if not (os.path.exists(path_img)):
            os.mkdir(path_img)
        file_img_path = path_img + "/" + code + "-" + datetime.datetime.now().strftime("%Y%m%d%H%M%S%f") + ".jpg"
        cv2.imwrite(str(file_img_path), image)
        return 0
    except:
        logging.error(f"Save_image: {traceback.format_exc()}")


# a function to save image when staff is checkin


# function insert info to database
def insert_information( code, accuracy):
    try:
        database = connect_database()
        def has_existed(cursor, code):
            cursor.execute(f"""select id from attendance_history_tbl
                            where code= '{code}' and
                                cast(time as date) = cast(GETDATE() as date)
                                order by time desc
                            """)
            result = cursor.fetchall()
            try:
                if len(result) == 0:
                    return None, False
                else:
                    return result[0][0], True
            except:
                logging.error(f"has_existed:{traceback.format_exc()}")
                return None, False

        cursor = database.cursor()
        uid, is_existed = has_existed(cursor, code)
        if is_existed is False:
            query = f"""
                insert into 
                    attendance_history_tbl(code,time,created_by,modified_by)
                values
                ('{code}',FORMAT(GETDATE(),'yyyy-MM-dd'),FORMAT(GETDATE(),'HH:mm'),'')
                select FORMAT(GETDATE(),'yyyy-MM-dd')
                                """

        else:
            query = f"""update attendance_history_tbl
                        set modified_by = FORMAT(GETDATE(),'HH:mm')
                        where id={uid};
                                """

        cursor.execute(query)
        database.commit()
        logging.info(f"Database: {code}")
        del query
        return 0
    except:
        logging.error(f"Insert_information: {traceback.format_exc()}")
        insert_infor(code,accuracy)


"""------------------------------------------------------------------"""


# a function to save image when staff is checkin
def has_existed_checkin(database, ma_nhan_vien):
    a = database.execute(
        f"SELECT id from checkin WHERE date(datetime) == date('now','localtime') and ma_nhan_vien == '{ma_nhan_vien}'")
    return list(a)


# a function to save image when staff is checkin
def has_existed_checkout(database, ma_nhan_vien):
    a = database.execute(
        f"SELECT id from checkout WHERE date(datetime) == date('now','localtime') and ma_nhan_vien == '{ma_nhan_vien}'")
    return list(a)


# function insert info to database
def insert_infor(ma_nhan_vien, accuracy):
    """
    code: string
    accuracy: float
    """
    path = "sources//database//data_base.sql"
    database = sqlite3.connect(path)
    infor_checkout = has_existed_checkout(database, ma_nhan_vien)
    infor_checkin = has_existed_checkin(database, ma_nhan_vien)
    if len(infor_checkin) == 0:
        query = f"""
                        INSERT INTO checkin(ma_nhan_vien,accuracy,datetime,image_path) 
                        VALUES('{ma_nhan_vien}','{accuracy}',datetime('now', 'localtime'),'image_path')
                    """

    elif len(infor_checkout) == 0:
        query = f"""
                        INSERT INTO checkout(ma_nhan_vien,accuracy,datetime,image_path) 
                        VALUES('{ma_nhan_vien}','{accuracy}',datetime('now', 'localtime'),'image_path')
                    """

    elif len(infor_checkout) == 1:
        query = f"""
                        UPDATE checkout 
                        SET datetime = datetime('now','localtime'),
                        accuracy = '{accuracy}', image_path = 'image_path' 
                        WHERE id = {infor_checkout[0][0]}
                    """

    database.execute(query)
    database.commit()
    database.close()
    del path, database, query
    return 0
# database = connect_database()
# a = take_information_database(database, "VTCODE02401")
# print(a)
# insert_infor("VTCODE02402",0.21)

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

im


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
def insert_information(ma_nhan_vien, accuracy):
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

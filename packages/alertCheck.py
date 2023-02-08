"""
author : Xuan Hoang
Data : 23/11/2022
Function : find matching face in faces


"""

import datetime

import pyttsx3

from packages.insert_information import connect_database


class AlertCheck:
    def __init__(self):
        self.engine = pyttsx3.init()
        # voices = engine.getProperty('voices')
        vi_voice_id = "HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\MSTTS_V110_viVN_An"
        self.engine.setProperty("voice", vi_voice_id)
        self.engine.setProperty("rate", 200)
    @staticmethod
    def take_information_database(code):
        """
        input: database and code to extract name from code
        output: name from code
        """
        database = connect_database()
        cursor = database.cursor()
        cursor.execute(f"""select name from employee_tbl where code ='{code}'""")
        try:
            return cursor.fetchone()[0]
        except:
            return " Người lạ "

    def alert(self, code):
        name = self.take_information_database(code)
        if datetime.datetime.now().hour > 12:
            speech = " Tạm biệt " + str(name)
        else:
            speech = " Xin chào " + str(name)
        self.engine.say(speech)
        self.engine.runAndWait()
        return 0



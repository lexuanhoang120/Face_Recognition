
from insert_information import *

ABSOLUTE_DIR = 'C:/Users/VTCODE/Documents/identify_human/sources/dataset/'
database = connect_database()
information_VTCODE = """ VTCODE00699,	Trương Nam Phước,	Giám đốc
VTCODE02352,	Phạm Vĩnh Toàn,	IT Manager
VTCODE2407,	Mai Trà My,	HCNS
VTCODE00384,	Hoàng Quốc Việt,	Technical Lead
VTCODE00351,	Lê Thị Dịu	,BA/QC Lead
VTCODE00389,	Nguyễn Xuân Vi	,Kỹ sư
VTCODE02338	,Nguyễn Thái Sang,	Frontend Lead
VTCODE02349,	Đặng Hoàng Phúc,	Frontend
VTCODE02353,	Phạm Anh Duy 	,Backend
VTCODE02354	,Bùi Đông Nhật	,Designer
VTCODE02367	,Nguyễn Văn Công	,SubLead Frontend
VTCODE02235	,Lê Tuấn Anh,	Backend Lead
VTCODE02385	,Nguyễn Phước Ngọc Ánh,	Tester
VTCODE02386	,Lê Thị Ngọc Giàu,	Tester
VTCODE02387	,Phạm Nguyễn Đức Duy	,Frontend
VTCODE02397	,Lê Khánh Phương	,Backend
VTCODE02398	,Trần Nguyên Khánh,	Backend
VTCODE02403	,Trần Phương Nguyên, 	BA
VTCODE02401	,Huỳnh Quang Duy	,Data Science
VTCODE02402	,Lê Xuân Hoằng 	,Data Science
VTCODE00473	,Trần Thanh Hải	,Bảo Trì (Lead)
VTCODE02029	,Trần Thanh Dũng	,R&D Lead
VTCODE02223	,Quan Văn Sang	,Bảo Trì
VTCODE02227	,Nguyễn Minh Thắng,	R&D
VTCODE02284	,Võ Mạnh Hùng	,Bảo Trì
VTCODE02379	,Nguyễn Minh Trọng,	R&D
VTCODE02172	,Nguyễn Ngọc Huy	,Parttime - Lập Trình
VTCODE2416	,Đỗ Nguyễn Tài Cẩn,	Frontend
VTCODE2418	,Bùi Thị Minh Thư,	BA
VTCODE2426	,Đoàn Như Dũng	,Data 
VTCODE-TTS-BA02	,Trần Thị Thu Kiều,	TTS-BA
VTCODE2429	,Nguyễn Minh Trân,	Tester
VTCODE-TTS-BA01	,Trần Thị Cẩm Giang,	TTS-BA
VTCODE-TTS-FE02	,Trần Thị Ngọc Thùy,	TTS-Frontend
VTCODE-TTS-FE03	,Hoàng Nhật Hiệp	,TTS-Frontend
VTCODE-TTS-DS02	,Trần Thùy Trang	,TTS-DS
VTCODE-TTS-HR01	,Phạm Thị Khánh Vy	,TTS-HCNS
VTCODE-TTS-XD01	,Nguyễn Đình Phúc,	TTS .NET
VTCO02410	,Huỳnh Đức Trung	,Kỹ sư thiết kế
VTCO00537,	Nguyễn Đức Anh	,Trưởng phòng thiết kế
VTCO02201,	Bạch Đức Thiện	,Kỹ sư thiết kế
VTCO02275,	Tôn Thất Trọng	,Kỹ sư thiết kế
VTCO02103	,Nguyễn Trường An,	Kỹ sư thiết kế
VTCO02419	,Vương Thị Nhã Phương,	Nhân viên hành chánh
VTCODE-TTS-BA03,	Phạm Hoàng Lan Chi	,TTS-BA"""
cursor = database.cursor()
for line in information_VTCODE.splitlines():

    code,name,position = line.split(',')
    code,name,position= code.strip(), name.strip(), position.strip()
    path_img = ABSOLUTE_DIR + code
    if not (os.path.exists(path_img)):
        os.mkdir(path_img)
    print(code,name,position)
    # query = f"""
    # insert into employee_tbl(code,name,position )
    # values
    # ('{code}',N'{name}',N'{position}')
    # """
    # cursor.execute(query)
    # database.commit()


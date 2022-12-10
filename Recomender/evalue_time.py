
import sys
sys.path.append('c:\\Users\\quach\\Desktop\\Personal\\FPT University\\SEMESTER 9\\Arranging-rooms-based-on-student-information')
from Recomender.Rec_main import RS
import time
st = time.time()
data = pd.read_csv("C:\\Users\\quach\\Desktop\\Personal\\FPT University\\SEMESTER 9\\Dataset\\FINAL_Data_set_FixHW.csv", encoding='utf-8')
RS = RS(data)
res = RS.arrange_ROOM()
res.to_csv("Result\\Room_result.csv",index = False)
et = time.time()
# get the execution time
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
print("FINISH")
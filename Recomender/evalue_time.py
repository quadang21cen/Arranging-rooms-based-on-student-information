
import sys
sys.path.append('c:\\Users\\quach\\Desktop\\Personal\\FPT University\\SEMESTER 9\\Arranging-rooms-based-on-student-information')
from Recomender.Rec_main import RS
import timeit
start = timeit.timeit()
RS = RS(path = '''C:\\Users\\quach\\Desktop\\Personal\\FPT University\\SEMESTER 9\\Arranging-rooms-based-on-student-information\\pre_processing\\Data_Augmentation_3k.csv''')
res = RS.arrange_ROOM()
end = timeit.timeit()
print(end - start)
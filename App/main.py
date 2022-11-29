#Thêm thư viện tkinter
from tkinter import *
import tkinter.font as tkFont



#Tạo một cửa sổ mới
window = Tk()

#Thêm tiêu đề cho cửa sổ
window.title('Welcome to DEMO')

#Đặt kích thước của cửa sổ
window.geometry('350x200')

#Lặp vô tận để hiển thị cửa sổ
window.mainloop()

# import tkinter as tk
# import tkinter.font as tkFont
#
# class App:
#     def __init__(self, root):
#         #setting title
#         root.title("undefined")
#         #setting window size
#         width=600
#         height=500
#         screenwidth = root.winfo_screenwidth()
#         screenheight = root.winfo_screenheight()
#         alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
#         root.geometry(alignstr)
#         root.resizable(width=False, height=False)
#
#         GLabel_305=tk.Label(root)
#         ft = tkFont.Font(family='Times',size=10)
#         GLabel_305["font"] = ft
#         GLabel_305["fg"] = "#333333"
#         GLabel_305["justify"] = "center"
#         GLabel_305["text"] = "Nhập thông tin dưới đây"
#         GLabel_305.place(x=170,y=60,width=218,height=30)
#
#         GLabel_233=tk.Label(root)
#         GLabel_233["bg"] = "#6d3030"
#         ft = tkFont.Font(family='Times',size=10)
#         GLabel_233["font"] = ft
#         GLabel_233["fg"] = "#333333"
#         GLabel_233["justify"] = "center"
#         GLabel_233["text"] = ""
#         GLabel_233.place(x=60,y=120,width=437,height=306)
#
#         GButton_648=tk.Button(root)
#         GButton_648["bg"] = "#f0f0f0"
#         ft = tkFont.Font(family='Times',size=10)
#         GButton_648["font"] = ft
#         GButton_648["fg"] = "#000000"
#         GButton_648["justify"] = "center"
#         GButton_648["text"] = "Enter"
#         GButton_648.place(x=380,y=160,width=70,height=25)
#         GButton_648["command"] = self.GButton_648_command
#
#     def GButton_648_command(self):
#         print("command")
#
# if __name__ == "__main__":
#     root = tk.Tk()
#     app = App(root)
#     root.mainloop()

# import tkinter as tk
# import tkinter.font as tkFont

# class App:
#     def __init__(self, root):
#         #setting title
#         root.title("undefined")
#         #setting window size
#         width=400
#         height=228
#         screenwidth = root.winfo_screenwidth()
#         screenheight = root.winfo_screenheight()
#         alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
#         root.geometry(alignstr)
#         root.resizable(width=False, height=False)

#         GLabel_448=tk.Label(root)
#         ft = tkFont.Font(family='Times',size=10)
#         GLabel_448["font"] = ft
#         GLabel_448["fg"] = "#333333"
#         GLabel_448["justify"] = "center"
#         GLabel_448["text"] = "label"
#         GLabel_448.place(x=20,y=20,width=70,height=25)

#         GButton_238=tk.Button(root)
#         GButton_238["bg"] = "#f0f0f0"
#         ft = tkFont.Font(family='Times',size=10)
#         GButton_238["font"] = ft
#         GButton_238["fg"] = "#000000"
#         GButton_238["justify"] = "center"
#         GButton_238["text"] = "Button"
#         GButton_238.place(x=140,y=130,width=70,height=25)
#         GButton_238["command"] = self.GButton_238_command

#         GLabel_885=tk.Label(root)
#         ft = tkFont.Font(family='Times',size=10)
#         GLabel_885["font"] = ft
#         GLabel_885["fg"] = "#333333"
#         GLabel_885["justify"] = "center"
#         GLabel_885["text"] = "label"
#         GLabel_885.place(x=20,y=60,width=70,height=25)

#         GLabel_117=tk.Label(root)
#         ft = tkFont.Font(family='Times',size=10)
#         GLabel_117["font"] = ft
#         GLabel_117["fg"] = "#333333"
#         GLabel_117["justify"] = "center"
#         GLabel_117["text"] = "label"
#         GLabel_117.place(x=260,y=20,width=70,height=25)

#         GLabel_221=tk.Label(root)
#         ft = tkFont.Font(family='Times',size=10)
#         GLabel_221["font"] = ft
#         GLabel_221["fg"] = "#333333"
#         GLabel_221["justify"] = "center"
#         GLabel_221["text"] = "label"
#         GLabel_221.place(x=260,y=50,width=70,height=25)

#     def GButton_238_command(self):
#         print("command")

# if __name__ == "__main__":
#     root = tk.Tk()
#     app = App(root)
#     root.mainloop()

#Thêm thư viện tkinter
from tkinter import *

#Tạo một cửa sổ mới
window = Tk()
window.title('Demo')
window.geometry('700x700')

# exe even when click
def clicked():
    lbl.configure(text="Button was clicked !!")

#Tạo một Textbox
txt = Entry(window)
#Vị trí xuất hiện của Textbox
txt.grid(column=1, row=0)
txt.focus()


#Thêm label có nội dung Hello, font chữ
lbl = Label(window, text="Text:")
lbl.grid(column=0, row=0)

#Thêm label có nội dung Hello, font chữ
lbl = Label(window, text="Cleaness and Privacy:")
lbl.grid(column=0, row=1)
#Tạo một Textbox
txt = Entry(window)
#Vị trí xuất hiện của Textbox
txt.grid(column=1, row=1)
txt.focus()

lbl = Label(window, text="Cleaness and Privacy:")
lbl.grid(column=0, row=1)
#Tạo một Textbox
txt = Entry(window)
#Vị trí xuất hiện của Textbox
txt.grid(column=1, row=1)
txt.focus()

#Thêm một nút nhấn Click Me

btn = Button(window, text="Choose File", bg="orange", fg="red", command=clicked)
btn.grid(column=1, row=3)

#Đặt kích thước của cửa sổ

#Lặp vô tận để hiển thị cửa sổ
window.mainloop()



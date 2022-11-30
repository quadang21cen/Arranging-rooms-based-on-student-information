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
import tkinter.font as tkFont
import customtkinter

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"



#Tạo một cửa sổ mới
window = customtkinter.CTk()
window.title('Demo')
window.geometry('700x700')

# exe even when click
def clicked():
    lbl.configure(text="Button was clicked !!")

#Tạo một Textbox
txt = customtkinter.CTkEntry(window)
#Vị trí xuất hiện của Textbox
txt.grid(column=1, row=0)
txt.focus()


#Thêm label có nội dung Hello, font chữ
lbl = customtkinter.CTkLabel(window, text="Text:")
lbl.grid(column=0, row=0)

#Thêm label có nội dung Hello, font chữ
lbl = customtkinter.CTkLabel(window, text="Cleaness and Privacy:")
lbl.grid(column=0, row=1)
#Tạo một Textbox
txt = customtkinter.CTkEntry(window)
#Vị trí xuất hiện của Textbox
txt.grid(column=1, row=1)
txt.focus()

lbl = customtkinter.CTkLabel(window, text="Cleaness and Privacy:")
lbl.grid(column=0, row=1)
#Tạo một Textbox
txt = customtkinter.CTkEntry(window)
#Vị trí xuất hiện của Textbox
txt.grid(column=1, row=1)
txt.focus()
print(value)
if value <= 1.5:
    res = "Red"
elif value <= 2.5:
    res = "Green"
elif value <= 3:
    res = "Blue"
else:
    res = "Invalid option"
lbl2.configure(text=res)
res = "Empty"
def viewSelected(value):
    glo

lbl2 = customtkinter.CTkLabel(window, text=str(res))
lbl2.grid(column=0, row=2)
tn = customtkinter.CTkSlider(window, orient='horizontal', from_=1, to=3, command=viewSelected)
tn.grid(column=1, row=2)
tn.focus()
#print(res)

#Thêm một nút nhấn Click Me

btn = customtkinter.CTkButton(window, text="Choose File", bg="orange", command=clicked)
btn.grid(column=1, row=3)


#Đặt kích thước của cửa sổ

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

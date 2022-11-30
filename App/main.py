# # import tkinter as tk
# # import tkinter.font as tkFont

# # class App:
# #     def __init__(self, root):
# #         #setting title
# #         root.title("undefined")
# #         #setting window size
# #         width=400
# #         height=228
# #         screenwidth = root.winfo_screenwidth()
# #         screenheight = root.winfo_screenheight()
# #         alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
# #         root.geometry(alignstr)
# #         root.resizable(width=False, height=False)

# #         GLabel_448=tk.Label(root)
# #         ft = tkFont.Font(family='Times',size=10)
# #         GLabel_448["font"] = ft
# #         GLabel_448["fg"] = "#333333"
# #         GLabel_448["justify"] = "center"
# #         GLabel_448["text"] = "label"
# #         GLabel_448.place(x=20,y=20,width=70,height=25)

# #         GButton_238=tk.Button(root)
# #         GButton_238["bg"] = "#f0f0f0"
# #         ft = tkFont.Font(family='Times',size=10)
# #         GButton_238["font"] = ft
# #         GButton_238["fg"] = "#000000"
# #         GButton_238["justify"] = "center"
# #         GButton_238["text"] = "Button"
# #         GButton_238.place(x=140,y=130,width=70,height=25)
# #         GButton_238["command"] = self.GButton_238_command

# #         GLabel_885=tk.Label(root)
# #         ft = tkFont.Font(family='Times',size=10)
# #         GLabel_885["font"] = ft
# #         GLabel_885["fg"] = "#333333"
# #         GLabel_885["justify"] = "center"
# #         GLabel_885["text"] = "label"
# #         GLabel_885.place(x=20,y=60,width=70,height=25)

# #         GLabel_117=tk.Label(root)
# #         ft = tkFont.Font(family='Times',size=10)
# #         GLabel_117["font"] = ft
# #         GLabel_117["fg"] = "#333333"
# #         GLabel_117["justify"] = "center"
# #         GLabel_117["text"] = "label"
# #         GLabel_117.place(x=260,y=20,width=70,height=25)

# #         GLabel_221=tk.Label(root)
# #         ft = tkFont.Font(family='Times',size=10)
# #         GLabel_221["font"] = ft
# #         GLabel_221["fg"] = "#333333"
# #         GLabel_221["justify"] = "center"
# #         GLabel_221["text"] = "label"
# #         GLabel_221.place(x=260,y=50,width=70,height=25)

# #     def GButton_238_command(self):
# #         print("command")

# # if __name__ == "__main__":
# #     root = tk.Tk()
# #     app = App(root)
# #     root.mainloop()

# #Thêm thư viện tkinter
# from tkinter import *
# import tkinter.font as tkFont



# #Tạo một cửa sổ mới
# window = Tk()
# window.title('Demo')
# window.geometry('700x700')

# # exe even when click
# def clicked():
#     lbl.configure(text="Button was clicked !!")

# #Tạo một Textbox
# txt = Entry(window)
# #Vị trí xuất hiện của Textbox
# txt.grid(column=1, row=0)
# txt.focus()


# #Thêm label có nội dung Hello, font chữ
# lbl = Label(window, text="Text:")
# lbl.grid(column=0, row=0)

# #Thêm label có nội dung Hello, font chữ
# lbl = Label(window, text="Cleaness and Privacy:")
# lbl.grid(column=0, row=1)
# #Tạo một Textbox
# txt = Entry(window)
# #Vị trí xuất hiện của Textbox
# txt.grid(column=1, row=1)
# txt.focus()

# lbl = Label(window, text="Cleaness and Privacy:")
# lbl.grid(column=0, row=1)
# #Tạo một Textbox
# txt = Entry(window)
# #Vị trí xuất hiện của Textbox
# txt.grid(column=1, row=1)
# txt.focus()

# #Thêm một nút nhấn Click Me

# btn = Button(window, text="Choose File", bg="orange", fg="red", command=clicked)
# btn.grid(column=1, row=3)

# #Đặt kích thước của cửa sổ

# #Lặp vô tận để hiển thị cửa sổ
# window.mainloop()

import customtkinter
from PIL import Image, ImageTk
import os
from tkinter import filedialog as fd

PATH = os.path.dirname(os.path.realpath(__file__))

customtkinter.set_appearance_mode("System")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("720x500")
        self.title("CustomTkinter example_button_images.py")

        self.grid_rowconfigure(2, weight=1)
        self.grid_columnconfigure(0, weight=1, minsize=200)

        self.frame_1 = customtkinter.CTkFrame(master=self, width=250, height=240, corner_radius=15)
        self.frame_1.grid(row=3, column=0, padx=20, pady=20, sticky="nsew")
        self.frame_1.grid_columnconfigure(0, weight=1)
   
        self.settings_image = self.load_image("/test_images/settings.png", 20)
        self.save_image = self.load_image("/test_images/save.png", 20)
        self.bell_image = self.load_image("/test_images/bell.png", 20)
        self.add_folder_image = self.load_image("/test_images/add-folder.png", 20)
        self.add_list_image = self.load_image("/test_images/add-folder.png", 20)
        self.add_user_image = self.load_image("/test_images/add-user.png", 20)
        self.chat_image = self.load_image("/test_images/chat.png", 20)
        self.home_image = self.load_image("/test_images/home.png", 20)


        entry_1 = customtkinter.CTkEntry(placeholder_text="CTkEntry")
        # entry_1.pack(pady=12, padx=10)
        # entry_1.grid(row=0, column=0,columnspan=1, padx=20, pady=(20, 10))
        entry_1.grid(row=0, column=0, columnspan=2, pady=20, padx=20, sticky="we")

        self.button_1 = customtkinter.CTkButton(master=self.frame_1, image=self.add_folder_image, text="Add Folder", height=32,
                                                compound="right", command=self.fun_BT1)
        self.button_1.grid(row=0, column=0, columnspan=1, padx=20, pady=(20, 10), sticky="ew")

        self.button_2 = customtkinter.CTkButton(master=self.frame_1, image=self.add_list_image, text="Start", height=32,
                                                compound="right", fg_color="#D35B58", hover_color="#C77C78",
                                                command=self.fun_BT2)
        self.button_2.grid(row=0, column=1, columnspan=1, padx=20, pady=(20, 10), sticky="ew")

        self.button_3 = customtkinter.CTkButton(master=self.frame_1, image=self.save_image, text="save", height=32,
                                                compound="right", fg_color="#D35B58", hover_color="#C77C78",
                                                command=self.fun_BT3)
        self.button_3.grid(row=0, column=2, columnspan=1, padx=20, pady=(20, 10), sticky="ew")



    def load_image(self, path, image_size):
        """ load rectangular image with path relative to PATH """
        return ImageTk.PhotoImage(Image.open(PATH + path).resize((image_size, image_size)))

    def fun_BT1(self):
        filename = fd.askopenfilename()
        print("button pressed")
    def fun_BT2(self):
        print("start running")
    def fun_BT3(self):
        print("start running")
        files = [('All Files', '*.*'), 
             ('Python Files', '*.py'),
             ('Text Document', '*.txt')]
        file = fd.asksaveasfile(filetypes = files, defaultextension = files)
if __name__ == "__main__":
    app = App()
    app.mainloop()

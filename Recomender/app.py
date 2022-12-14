import tkinter
import tkinter.messagebox
import customtkinter
from PIL import Image, ImageTk
import os
from tkinter import filedialog as fd
from Rec_main import RS
import threading
import pandas as pd

customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

PATH = os.path.dirname(os.path.realpath(__file__))


class App(customtkinter.CTk):

    APP_NAME = "RS.py"
    WIDTH = 900
    HEIGHT = 600

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.corr_rs = df = pd.DataFrame()
        self.title(App.APP_NAME)
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.RS = None
        self.filename = None

        self.frame = customtkinter.CTkFrame(master=self,
                                            width=App.WIDTH,
                                            height=App.HEIGHT,
                                            corner_radius=0)
        self.frame.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

        self.frame_0 = customtkinter.CTkFrame(master=self.frame, width=650, height=270, corner_radius=15,
                                              fg_color=("white", "gray38"), )
        self.frame_0.place(relx=0.5, rely=0.3, anchor=tkinter.CENTER)
        self.frame_0.grid_columnconfigure(0, weight=1)
        self.frame_0.grid_columnconfigure(1, weight=1)

        self.label_1 = customtkinter.CTkLabel(master=self.frame_0, width=200, height=60,
                                               text="Choosing weight for recommender system")
        self.label_1.place(relx=0.5, rely=0.15, anchor=tkinter.CENTER)

        self.hobbyLabel = customtkinter.CTkLabel(master=self.frame_0,
                                              text="Hobby",width=50)
        self.hobbyLabel.place(relx=0.1, rely=0.45, anchor=tkinter.E)

        self.hobbyentry = customtkinter.CTkEntry(master=self.frame_0, corner_radius=6, width=100)
        self.hobbyentry.place(relx=0.27, rely=0.45, anchor=tkinter.CENTER)

        self.hometownLabel = customtkinter.CTkLabel(master=self.frame_0,
                                                 text="Hometown", width=50)
        self.hometownLabel.place(relx=0.6, rely=0.45, anchor=tkinter.CENTER)

        self.hometownentry = customtkinter.CTkEntry(master=self.frame_0, corner_radius=6, width=100)
        self.hometownentry.place(relx=0.8, rely=0.45, anchor=tkinter.W)


        self.cleanliness_privacyLabel = customtkinter.CTkLabel(master=self.frame_0,
                                                  text="Cleaniness and Privacy")
        self.cleanliness_privacyLabel.place(relx=0.6, rely=0.65, anchor=tkinter.CENTER)

        self.cleanliness_privacy_entry = customtkinter.CTkEntry(master=self.frame_0, corner_radius=6, width=100)
        self.cleanliness_privacy_entry.place(relx=0.8, rely=0.65, anchor=tkinter.W)

        self.foodLabel = customtkinter.CTkLabel(master=self.frame_0,
                                              text="Food",width=50)
        self.foodLabel.place(relx=0.1, rely=0.65, anchor=tkinter.E)

        self.foodentry = customtkinter.CTkEntry(master=self.frame_0, corner_radius=6, width=100)
        self.foodentry.place(relx=0.27, rely=0.65, anchor=tkinter.CENTER)

        self.personalityLabel = customtkinter.CTkLabel(master=self.frame_0,
                                                   text="Personality",width=50)
        self.personalityLabel.place(relx=0.115, rely=0.85, anchor=tkinter.E)

        self.personalityentry = customtkinter.CTkEntry(master=self.frame_0, corner_radius=6, width=100)
        self.personalityentry.place(relx=0.27, rely=0.85, anchor=tkinter.CENTER)

        self.refLabel = customtkinter.CTkLabel(master=self.frame_0,
                                              text="Reference",width=50)
        self.refLabel.place(relx=0.6, rely=0.85, anchor=tkinter.CENTER)

        self.refentry = customtkinter.CTkEntry(master=self.frame_0, corner_radius=6, width=100)
        self.refentry.place(relx=0.8, rely=0.85, anchor=tkinter.W)

        self.genderLabel = customtkinter.CTkLabel(master=self.frame,
                                                  text="Gender Separation?")
        self.genderLabel.place(relx=0.2, rely=0.6, anchor=tkinter.E)

        self.gender_switch = customtkinter.CTkSwitch(master=self.frame, text="No/Yes")
        self.gender_switch.place(relx=0.3, rely=0.6, anchor=tkinter.CENTER)

        self.roomLabel = customtkinter.CTkLabel(master=self.frame,
                                                    text="Number of people", width=50)
        self.roomLabel.place(relx=0.17, rely=0.65, anchor=tkinter.E)

        self.radio_var = tkinter.IntVar(value=0)
        self.room2Radio = customtkinter.CTkRadioButton(master=self.frame,
                                                           variable=self.radio_var,
                                                           value=2, text = "2")
        self.room2Radio.place(relx=0.27, rely=0.65, anchor=tkinter.E)

        self.room3Radio = customtkinter.CTkRadioButton(master=self.frame,
                                                      variable=self.radio_var,
                                                      value=3, text="3")
        self.room3Radio.place(relx=0.37, rely=0.65, anchor=tkinter.E)

        self.room4Radio = customtkinter.CTkRadioButton(master=self.frame,
                                                       variable=self.radio_var,
                                                       value=4, text="4")
        self.room4Radio.place(relx=0.47, rely=0.65, anchor=tkinter.E)

        self.room5Radio = customtkinter.CTkRadioButton(master=self.frame,
                                                       variable=self.radio_var,
                                                       value=5, text="5")
        self.room5Radio.place(relx=0.57, rely=0.65, anchor=tkinter.E)

        self.room6Radio = customtkinter.CTkRadioButton(master=self.frame,
                                                       variable=self.radio_var,
                                                       value=6, text="6")
        self.room6Radio.place(relx=0.67, rely=0.65, anchor=tkinter.E)

        self.room7Radio = customtkinter.CTkRadioButton(master=self.frame,
                                                       variable=self.radio_var,
                                                       value=7, text="7")
        self.room7Radio.place(relx=0.77, rely=0.65, anchor=tkinter.E)

        self.room8Radio = customtkinter.CTkRadioButton(master=self.frame,
                                                       variable=self.radio_var,
                                                       value=8, text="8")
        self.room8Radio.place(relx=0.87, rely=0.65, anchor=tkinter.E)

        self.room9Radio = customtkinter.CTkRadioButton(master=self.frame,
                                                       variable=self.radio_var,
                                                       value=9, text="9")
        self.room9Radio.place(relx=0.97, rely=0.65, anchor=tkinter.E)


        self.sliderLabel = customtkinter.CTkLabel(master=self.frame,
                                                  text="Constrast",width=50)
        self.sliderLabel.place(relx=0.17, rely=0.75, anchor=tkinter.E)

        self.slider_value = 0
        self.slider_contrast = customtkinter.CTkSlider(master=self.frame, orient='horizontal', from_=0, to=100, number_of_steps=100, command=self.slider_value_get, width= 500)
        self.slider_contrast.place(relx=0.55, rely=0.75, anchor=tkinter.CENTER)

        self.progressbar = customtkinter.CTkProgressBar(master=self.frame, width=550)
        self.progressbar.place(relx=0.5, rely=0.8, anchor=tkinter.CENTER)

        self.frame_1 = customtkinter.CTkFrame(master=self.frame, width=250, height=50, corner_radius=15, fg_color=("white", "gray38"),)
        self.frame_1.place(relx=0.5, rely=0.9, anchor=tkinter.CENTER)
        self.frame_1.grid_columnconfigure(0, weight=1)
        self.frame_1.grid_columnconfigure(1, weight=1)

        self.settings_image = self.load_image("\\test_images\\settings.png", 20)
        self.save_image = self.load_image("\\test_images\\save.png", 20)
        self.bell_image = self.load_image("\\test_images\\bell.png", 20)
        self.add_folder_image = self.load_image("\\test_images\\add-folder.png", 20)
        self.add_list_image = self.load_image("\\test_images\\add-folder.png", 20)
        self.add_user_image = self.load_image("\\test_images\\add-user.png", 20)
        self.chat_image = self.load_image("\\test_images\\chat.png", 20)
        self.home_image = self.load_image("\\test_images\\home.png", 20)

        self.button_1 = customtkinter.CTkButton(master=self.frame_1, image=self.add_folder_image, text="Add Folder",
                                                height=32,
                                                compound="right", command=self.fun_BT1)
        self.button_1.grid(row=0, column=0, columnspan=1, padx=20, pady=(20, 10), sticky="ew")

        self.t = threading.Thread(target=self.fun_BT2)
        self.stop_thread = False
        self.button_2 = customtkinter.CTkButton(master=self.frame_1, image=self.add_list_image, text="Start", height=32,
                                                compound="right", fg_color="#D35B58", hover_color="#C77C78",
                                                command=self.stat_thread_fun_BT2)
        self.button_2.grid(row=0, column=1, columnspan=1, padx=20, pady=(20, 10), sticky="ew")

        self.button_3 = customtkinter.CTkButton(master=self.frame_1, image=self.save_image, text="save", height=32,
                                                compound="right", fg_color="#D35B58", hover_color="#C77C78",
                                                command=self.fun_BT3)
        self.button_3.grid(row=0, column=2, columnspan=1, padx=20, pady=(20, 10), sticky="ew")


    def load_image(self, path, image_size):
        """ load rectangular image with path relative to PATH """
        return ImageTk.PhotoImage(Image.open(PATH + path).resize((image_size, image_size)))
    def fun_BT1(self):

        self.filename = fd.askopenfilename()
        #print(self.filename)
        print("button pressed")
    def stat_thread_fun_BT2(self):
        if self.t.is_alive():
            tkinter.messagebox.showerror('Program is running', 'Error: The program is running. Please Stand By !')
            return
        self.t = threading.Thread(target=self.fun_BT2)
        self.t.start()
        stop_event = threading.Event()
        stop_event.set()
        


    def fun_BT2(self):
        if self.filename == None:
            tkinter.messagebox.showwarning('Missing file path', 'This button works if there is a file !')
            return
        self.RS = RS(df_path = self.filename)
        # self.corr_rs = self.RS.compute_all_corr()
        W_hob = self.hobbyentry.get()
        food_value = self.foodentry.get()
        W_Bio_per = self.personalityentry.get()
        W_hom = self.hometownentry.get()
        W_cp = self.cleanliness_privacy_entry.get()
        W_ref = self.refentry.get()
        W_food = self.foodentry.get()
        split_gender = self.gender_switch.get() # Value 0 for not or 1 for yes
        contrast_value = float(self.slider_contrast.get()/100)
        room_size = self.radio_var.get()
        ls_weight = [W_hom,W_Bio_per,W_food, W_hob, W_ref,W_cp, room_size, contrast_value]
        number_empty = []
        for i in range(len(ls_weight)):
            if len(ls_weight[i]) == 0:
                number_empty.append(i)
        for j in ls_weight:
            ls_weight[i] = 100/(len(number_empty))
        self.corr_rs = self.RS.arrange_ROOM(ls_weight,split_gender = split_gender)
        tkinter.messagebox.showinfo('Program done', 'FINISH COMPUTE CORR !')
        print("FINISH COMPUTE CORR")
    def fun_BT3(self):
        if self.corr_rs.empty:
            tkinter.messagebox.showwarning('Need to run first', 'There are no file to save !')
            return
        print("start running")
        files = [('All Files', '*.*'),
             ('Python Files', '*.py'),
             ('Text Document', '*.txt')]
        file = fd.asksaveasfile(filetypes = files, defaultextension = files)
        self.corr_rs.to_csv(file)
    def slider_value_get(self, val):
        self.slider_value = val
    def on_closing(self, event=0):
        self.destroy()

    def start(self):
        self.mainloop()


if __name__ == "__main__":
    app = App()
    app.start()

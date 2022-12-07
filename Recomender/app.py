import tkinter
import tkinter.messagebox
import customtkinter
from PIL import Image, ImageTk
import os
from tkinter import filedialog as fd
from Rec_main import RS

customtkinter.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
customtkinter.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

PATH = os.path.dirname(os.path.realpath(__file__))


class App(customtkinter.CTk):

    APP_NAME = "CustomTkinter example_background_image.py"
    WIDTH = 900
    HEIGHT = 600

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.title(App.APP_NAME)
        self.geometry(f"{App.WIDTH}x{App.HEIGHT}")

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.RS = None

        self.frame = customtkinter.CTkFrame(master=self,
                                            width=App.WIDTH,
                                            height=App.HEIGHT,
                                            corner_radius=0)
        self.frame.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

        self.frame_0 = customtkinter.CTkFrame(master=self.frame, width=650, height=270, corner_radius=15,
                                              fg_color=("white", "gray38"), )
        self.frame_0.place(relx=0.5, rely=0.33, anchor=tkinter.CENTER)
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
                                              text="food",width=50)
        self.foodLabel.place(relx=0.1, rely=0.65, anchor=tkinter.E)

        self.foodentry = customtkinter.CTkEntry(master=self.frame_0, corner_radius=6, width=100)
        self.foodentry.place(relx=0.27, rely=0.65, anchor=tkinter.CENTER)

        self.personalityLabel = customtkinter.CTkLabel(master=self.frame_0,
                                                   text="Personality",width=50)
        self.personalityLabel.place(relx=0.115, rely=0.85, anchor=tkinter.E)

        self.personalityentry = customtkinter.CTkEntry(master=self.frame_0, corner_radius=6, width=100)
        self.personalityentry.place(relx=0.27, rely=0.85, anchor=tkinter.CENTER)

        self.genderLabel = customtkinter.CTkLabel(master=self.frame_0,
                                                  text="Gender Separation?")
        self.genderLabel.place(relx=0.6, rely=0.85, anchor=tkinter.CENTER)

        self.gender_switch = customtkinter.CTkSwitch(master=self.frame_0, text="No/Yes")
        self.gender_switch.place(relx=0.8, rely=0.85, anchor=tkinter.W)

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

        self.settings_image = self.load_image("Recomender/test_images/settings.png", 20)
        self.save_image = self.load_image("Recomender/test_images/save.png", 20)
        self.bell_image = self.load_image("Recomender/test_images/bell.png", 20)
        self.add_folder_image = self.load_image("Recomender/test_images/add-folder.png", 20)
        self.add_list_image = self.load_image("Recomender/test_images/add-folder.png", 20)
        self.add_user_image = self.load_image("Recomender/test_images/add-user.png", 20)
        self.chat_image = self.load_image("Recomender/test_images/chat.png", 20)
        self.home_image = self.load_image("Recomender/test_images/home.png", 20)

        self.button_1 = customtkinter.CTkButton(master=self.frame_1, image=self.add_folder_image, text="Add Folder",
                                                height=32,
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
        self.filename = fd.askopenfilename()
        #print(self.filename)
        print("button pressed")
    def fun_BT2(self):
        self.RS = RS(path = self.filename)
        res = self.RS.compute_all_corr()

        # hobby_value = self.hobbyentry.get()
        # food_value = self.foodentry.get()
        # personality_value = self.personalityentry.get()
        # hometown_value = self.hometownentry.get()
        # cleanliness_privacy_value = self.cleanliness_privacy_entry.get()
        # gender_switch_value = self.gender_switch.get() # Value 0 or 1
        # contrast_value = float(self.slider_contrast.get()/100)
        # num_people_value = self.radio_var.get()
        print("start running")
    def fun_BT3(self):
        print("start running")
        files = [('All Files', '*.*'),
             ('Python Files', '*.py'),
             ('Text Document', '*.txt')]
        file = fd.asksaveasfile(filetypes = files, defaultextension = files)
    def slider_value_get(self, val):
        self.slider_value = val
    def on_closing(self, event=0):
        self.destroy()

    def start(self):
        self.mainloop()


if __name__ == "__main__":
    app = App()
    app.start()
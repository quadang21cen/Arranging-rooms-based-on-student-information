import tkinter
import tkinter.messagebox
import customtkinter
from PIL import Image, ImageTk
import os
from tkinter import filedialog as fd

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
        self.full = False


        self.frame = customtkinter.CTkFrame(master=self,
                                            width=700,
                                            height=App.HEIGHT,
                                            corner_radius=0)
        self.frame.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

        self.label_1 = customtkinter.CTkLabel(master=self.frame, width=200, height=60,
                                               text="Choosing weight for recommender system")
        self.label_1.place(relx=0.5, rely=0.1, anchor=tkinter.CENTER)

        self.hobbyLabel = customtkinter.CTkLabel(master=self.frame,
                                              text="Hobby",width=50)
        self.hobbyLabel.place(relx=0.1, rely=0.2, anchor=tkinter.E)

        self.hobbyentry = customtkinter.CTkEntry(master=self.frame, corner_radius=6, width=100)
        self.hobbyentry.place(relx=0.2, rely=0.2, anchor=tkinter.CENTER)

        self.hometownLabel = customtkinter.CTkLabel(master=self.frame,
                                                 text="Hometown", width=50)
        self.hometownLabel.place(relx=0.6, rely=0.2, anchor=tkinter.CENTER)

        self.hometownentry = customtkinter.CTkEntry(master=self.frame, corner_radius=6, width=100)
        self.hometownentry.place(relx=0.8, rely=0.2, anchor=tkinter.W)


        self.cleanliness_privacyLabel = customtkinter.CTkLabel(master=self.frame,
                                                  text="Cleaniness and Privacy")
        self.cleanliness_privacyLabel.place(relx=0.6, rely=0.3, anchor=tkinter.CENTER)

        self.cleanliness_privacy_entry = customtkinter.CTkEntry(master=self.frame, corner_radius=6, width=100)
        self.cleanliness_privacy_entry.place(relx=0.8, rely=0.3, anchor=tkinter.W)

        self.foodLabel = customtkinter.CTkLabel(master=self.frame,
                                              text="food",width=50)
        self.foodLabel.place(relx=0.1, rely=0.3, anchor=tkinter.E)

        self.foodentry = customtkinter.CTkEntry(master=self.frame, corner_radius=6, width=100)
        self.foodentry.place(relx=0.2, rely=0.3, anchor=tkinter.CENTER)

        self.personalityLabel = customtkinter.CTkLabel(master=self.frame,
                                                   text="Personality",width=50)
        self.personalityLabel.place(relx=0.1, rely=0.4, anchor=tkinter.E)

        self.personalityentry = customtkinter.CTkEntry(master=self.frame, corner_radius=6, width=100)
        self.personalityentry.place(relx=0.2, rely=0.4, anchor=tkinter.CENTER)

        self.genderLabel = customtkinter.CTkLabel(master=self.frame,
                                                  text="Gender Separation?")
        self.genderLabel.place(relx=0.6, rely=0.4, anchor=tkinter.CENTER)

        self.gender_switch = customtkinter.CTkSwitch(master=self.frame, text="Yes/No")
        self.gender_switch.place(relx=0.8, rely=0.4, anchor=tkinter.W)


        self.sliderLabel = customtkinter.CTkLabel(master=self.frame,
                                                  text="Constrast",width=50)
        self.sliderLabel.place(relx=0.1, rely=0.6, anchor=tkinter.E)

        self.slider_value = 0
        self.slider1 = customtkinter.CTkSlider(master=self.frame, orient='horizontal', from_=0, to=100, number_of_steps=100, command=self.slider_value_get, width= 400)
        self.slider1.place(relx=0.5, rely=0.6, anchor=tkinter.CENTER)

        self.frame_1 = customtkinter.CTkFrame(master=self, width=250, height=50, corner_radius=15, fg_color=("white", "gray38"),)
        self.frame_1.place(relx=0.5, rely=0.7, anchor=tkinter.CENTER)
        self.frame_1.grid_columnconfigure(0, weight=1)
        self.frame_1.grid_columnconfigure(1, weight=1)

        self.settings_image = self.load_image("/test_images/settings.png", 20)
        self.save_image = self.load_image("/test_images/save.png", 20)
        self.bell_image = self.load_image("/test_images/bell.png", 20)
        self.add_folder_image = self.load_image("/test_images/add-folder.png", 20)
        self.add_list_image = self.load_image("/test_images/add-folder.png", 20)
        self.add_user_image = self.load_image("/test_images/add-user.png", 20)
        self.chat_image = self.load_image("/test_images/chat.png", 20)
        self.home_image = self.load_image("/test_images/home.png", 20)

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

        # self.settings_image = self.load_image("/test_images/settings.png", 20)
        # self.save_image = self.load_image("/test_images/save.png", 20)
        # self.bell_image = self.load_image("/test_images/bell.png", 20)
        # self.add_folder_image = self.load_image("/test_images/add-folder.png", 20)
        # self.add_list_image = self.load_image("/test_images/add-folder.png", 20)
        # self.add_user_image = self.load_image("/test_images/add-user.png", 20)
        # self.chat_image = self.load_image("/test_images/chat.png", 20)
        # self.home_image = self.load_image("/test_images/home.png", 20)

        # self.button_2 = customtkinter.CTkButton(master=self.frame, text="Open File CSV",
        #                                         corner_radius=6, command=self.button_event, width=100)
        # self.button_2.place(relx=0.1, rely=0.7, anchor=tkinter.CENTER)
        #
        # self.button_3 = customtkinter.CTkButton(master=self.frame, text="Running",
        #                                         corner_radius=6, command=self.button_event, width=100)
        # self.button_3.place(relx=0.5, rely=0.7, anchor=tkinter.CENTER)
        #
        # self.button_4 = customtkinter.CTkButton(master=self.frame, text="Saving",
        #                                         corner_radius=6, command=self.button_event, width=100)
        # self.button_4.place(relx=0.9, rely=0.7, anchor=tkinter.CENTER)

    def button_event(self):
        print("Login pressed - username:", self.entry_1.get(), "password:", self.entry_2.get())
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
    def slider_value_get(self, val):
        self.slider_value = val
        print(self.slider_value)
        #print(self.slider_value)

    # def check(self, *args):
    #     if len(str(self.get())) <= 5:
    #         self.old_value = self.get() # accept change
    #     else:
    #         self.var.set(self.old_value) # reject change

    def on_closing(self, event=0):
        self.destroy()

    def start(self):
        self.mainloop()


if __name__ == "__main__":
    app = App()
    app.start()
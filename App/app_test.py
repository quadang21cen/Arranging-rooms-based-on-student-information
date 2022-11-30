import tkinter
import tkinter.messagebox
import customtkinter
from PIL import Image, ImageTk
import os

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
        self.minsize(App.WIDTH, App.HEIGHT)
        self.maxsize(App.WIDTH, App.HEIGHT)
        self.resizable(False, False)

        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.full = False


        self.frame = customtkinter.CTkFrame(master=self,
                                            width=700,
                                            height=App.HEIGHT,
                                            corner_radius=0)
        self.frame.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

        self.label_1 = customtkinter.CTkLabel(master=self.frame, width=200, height=60,
                                              fg_color=("gray70", "gray25"), text="Choosing weight for recommender system")
        self.label_1.place(relx=0.5, rely=0.1, anchor=tkinter.CENTER)

        self.hobbyLabel = customtkinter.CTkLabel(master=self.frame,
                                              fg_color=("gray70", "gray25"),
                                              text="Hobby")
        self.hobbyLabel.place(relx=0.2, rely=0.2, anchor=tkinter.CENTER)

        self.hobbyentry = customtkinter.CTkEntry(master=self.frame, corner_radius=6, width=100)
        self.hobbyentry.place(relx=0.5, rely=0.2, anchor=tkinter.CENTER)

        self.foodLabel = customtkinter.CTkLabel(master=self.frame,
                                              fg_color=("gray70", "gray25"),
                                              text="food")
        self.foodLabel.place(relx=0.2, rely=0.3, anchor=tkinter.CENTER)

        self.foodentry = customtkinter.CTkEntry(master=self.frame, corner_radius=6, width=100)
        self.foodentry.place(relx=0.5, rely=0.3, anchor=tkinter.CENTER)

        self.personalityLabel = customtkinter.CTkLabel(master=self.frame,
                                                   fg_color=("gray70", "gray25"),
                                                   text="Personality")
        self.personalityLabel.place(relx=0.2, rely=0.4, anchor=tkinter.CENTER)

        self.personalityentry = customtkinter.CTkEntry(master=self.frame, corner_radius=6, width=100)
        self.personalityentry.place(relx=0.5, rely=0.4, anchor=tkinter.CENTER)

        self.genderLabel = customtkinter.CTkLabel(master=self.frame,
                                                       fg_color=("gray70", "gray25"),
                                                       text="Gender Separation?")
        self.genderLabel.place(relx=0.2, rely=0.5, anchor=tkinter.CENTER)

        self.gender_switch = customtkinter.CTkSwitch(master=self.frame, text="Yes/No")
        self.gender_switch.place(relx=0.5, rely=0.5, anchor=tkinter.CENTER)

        self.sliderLabel = customtkinter.CTkLabel(master=self.frame,
                                                  fg_color=("gray70", "gray25"),
                                                  text="Slider")
        self.sliderLabel.place(relx=0.2, rely=0.6, anchor=tkinter.CENTER)

        self.slider_value = 0
        self.slider1 = customtkinter.CTkSlider(master=self.frame, orient='horizontal', from_=0, to=100, command=self.slider_value_get)
        self.slider1.place(relx=0.5, rely=0.6, anchor=tkinter.CENTER)

        self.button_2 = customtkinter.CTkButton(master=self.frame, text="Login",
                                                corner_radius=6, command=self.button_event, width=100)
        self.button_2.place(relx=0.1, rely=0.7, anchor=tkinter.CENTER)

        self.button_3 = customtkinter.CTkButton(master=self.frame, text="Login",
                                                corner_radius=6, command=self.button_event, width=100)
        self.button_3.place(relx=0.5, rely=0.7, anchor=tkinter.CENTER)

        self.button_4 = customtkinter.CTkButton(master=self.frame, text="Login",
                                                corner_radius=6, command=self.button_event, width=100)
        self.button_4.place(relx=0.9, rely=0.7, anchor=tkinter.CENTER)

    def button_event(self):
        print("Login pressed - username:", self.entry_1.get(), "password:", self.entry_2.get())

    def slider_value_get(self, val):
        self.slider_value = val
        print(self.slider_value)

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
import customtkinter
from PIL import Image, ImageTk
import os
from tkinter import filedialog as fd


class App(customtkinter.CTk):
    def __init__(self):
        super().__init__()
        self.geometry("720x500")
        self.title("CustomTkinter")
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(1, weight=1, minsize=200)

        entry_1 = customtkinter.CTkEntry(placeholder_text="CTkEntry")
        # entry_1.pack(pady=12, padx=10)
        # entry_1.grid(row=0, column=0,columnspan=1, padx=20, pady=(20, 10))
        entry_1.grid(row=0, column=0, columnspan=1, pady=20, padx=20)


        # self.frame_1 = customtkinter.CTkFrame(master=self, width=250, height=240, corner_radius=15)
        # self.frame_1.grid(row=3, column=0, padx=20, pady=20, sticky="nsew")
        # self.frame_1.grid_columnconfigure(0, weight=1)




if __name__ == "__main__":
    app = App()
    app.mainloop()

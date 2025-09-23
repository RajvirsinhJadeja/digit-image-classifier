import customtkinter as ctk


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Number Detector")
        window_width = 800
        window_height = 400
        
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        x = int((screen_width / 2) - (window_width / 2))
        y = int((screen_height / 2) - (window_height / 2))
        
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")
        
        scale = 10
        canvas = ctk.CTkCanvas(self, width=28*scale, height=28*scale, bg="black")
        canvas.grid(row=0, column=0)
        
        text_title = ctk.CTkLabel(self, text="The number you drew is", font=("Inter", 26))
        text_prediction = ctk.CTkLabel(self, text=f"{scale}", font=("Inter", 26))
        text_propability = ctk.CTkLabel(self, text="99 sure", font=("Inter", 26))
        text_title.grid(row=0, column=1)
        text_prediction.grid(row=1, column=1)
        text_propability.grid(row=2, column=1)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.grid_rowconfigure(2, weight=1)
        

app = App()
app.mainloop()
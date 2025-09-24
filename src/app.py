import customtkinter as ctk
from mood_neural_network import neuralNetwork
from PIL import ImageGrab
import cupy as cy


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.nn = neuralNetwork()
        self.nn.load_weights_biases()
        
        self.title("Number Detector")
        window_width = 650
        window_height = 360
        
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        
        x = int((screen_width / 2) - (window_width / 2))
        y = int((screen_height / 2) - (window_height / 2))
        
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.resizable(False, False)
        
        
        # ---------- Left Frame
        left_frame = ctk.CTkFrame(self)
        left_frame.pack(side="left", fill="both", expand=False, padx=10, pady=10)
        
        scale = 10
        self.canvas = ctk.CTkCanvas(left_frame, width=28*scale, height=28*scale, bg="black", highlightthickness=0, bd=0)
        self.canvas.pack(pady=(0, 10))
        
        self.old_x = None
        self.old_y = None
        
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.reset)
        
        clear_button = ctk.CTkButton(left_frame, text="Clear", command=self.clear_canvas)
        clear_button.pack(pady=(0, 10))
        
        # ---------- Right Frame
        right_frame = ctk.CTkFrame(self)
        right_frame.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        ctk.CTkLabel(right_frame, text="").pack(expand=True)

        self.prediction = ""
        self.prediction_percentage = ""
        
        self.label1 = ctk.CTkLabel(right_frame, text=f"{self.prediction}", font=("Inter", 40))
        self.label1.pack(pady=10)

        self.label2 = ctk.CTkLabel(right_frame, text=f"Confidence: {self.prediction_percentage}%", font=("Inter", 20))
        self.label2.pack(pady=10)
        
        ctk.CTkLabel(right_frame, text="").pack(expand=True)


    def start_draw(self, event):
        self.old_x, self.old_y = event.x, event.y

    
    def draw(self, event):
        x, y = event.x, event.y
        self.canvas.create_line(self.old_x, self.old_y, x, y, width=20, fill="white", capstyle="round", smooth=True) # type: ignore
        self.old_x, self.old_y = x, y


    def reset(self, event):
        self.old_x, self.old_y = None, None
        self.get_canvas_image()


    def clear_canvas(self):
        self.canvas.delete("all")
    
    
    def get_canvas_image(self):
        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        w = self.canvas.winfo_width()
        h = self.canvas.winfo_height()
        
        img = ImageGrab.grab((x, y, x+w, y+h))
        
        gray = img.convert("L")
        resized_gray = gray.resize((28, 28))
        resized_gray.save("canvas_image.png")
        pixels = cy.array(resized_gray.getdata())
        
        print(pixels)
        
        z_list, activation_list = self.nn.forward_pass(x=pixels, dropout_rate=0)
        self.prediction = cy.argmax(activation_list[-1])
        self.prediction_percentage = round(float(activation_list[-1][self.prediction]) * 100, 2)
        
        self.label1.configure(text=f"{self.prediction}")
        self.label2.configure(text=f"Confidence: {self.prediction_percentage}%")
        

app = App()
app.mainloop()
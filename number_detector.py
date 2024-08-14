import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageDraw
from tensorflow.keras.models import model_from_json
import numpy as np
import cv2 as cv
from datetime import datetime
import os

# loads Neural Network into project
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.weights.h5")


class NumberDetector:
    def __init__(self, parent, posx, posy):
        self.parent = parent
        self.posx = posx
        self.posy = posy
        self.sizex = 625
        self.sizey = 500
        self.b1 = "up"
        self.xold = None
        self.yold = None

        self.drawing_area = tk.Canvas(self.parent, width=self.sizex, height=self.sizey)
        self.drawing_area.place(x=self.posx, y=self.posy)
        self.drawing_area.bind("<Motion>", self.motion)
        self.drawing_area.bind("<ButtonPress-1>", self.b1down)
        self.drawing_area.bind("<ButtonRelease-1>", self.b1up)

        convert_clear = tk.Frame(self.parent)
        convert_clear.pack(side=tk.BOTTOM)
        self.button = tk.Button(convert_clear, text="Detect!", width=10, bg='white', command=self.prediction)
        self.button1 = tk.Button(convert_clear, text="Clear!", width=10, bg='white', command=self.clear)
        self.button.pack(side=tk.LEFT)
        self.button1.pack(side=tk.LEFT)

        self.image = Image.new("RGB", (625, 500), 'black')
        self.draw = ImageDraw.Draw(self.image)

    def save(self):
        current_date = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        filename = current_date + ".jpg"
        self.image.save('Images/' + filename)
        return 'Images/' + filename

    def prediction(self):
        # reads in the image and formats it to be predicted
        filename = self.save()
        image = cv.imread(filename)
        new_img = cv.resize(image, (28, 28))
        fin_img = cv.cvtColor(new_img, cv.COLOR_BGR2GRAY)
        np_image = np.asarray(fin_img) / 255
        os.remove(filename)

        # Predicts the number from the Image
        prediction = loaded_model.predict(np_image.reshape(1, 28, 28))
        output = np.argmax(prediction)
        messagebox.showinfo("Draw to Text", message="The number you wrote was: " + str(output))

    def clear(self):
        self.drawing_area.delete("all")
        self.image = Image.new("RGB", (620, 500), 'black')
        self.draw = ImageDraw.Draw(self.image)

    def b1down(self, event):
        self.b1 = "down"

    def b1up(self, event):
        self.b1 = "up"
        self.xold = None
        self.yold = None

    def motion(self, event):
        if self.b1 == "down":
            if self.xold is not None and self.yold is not None:
                event.widget.create_line(self.xold, self.yold, event.x, event.y, smooth='true', width=40, fill='black',
                                         capstyle='round')
                self.draw.line(((self.xold, self.yold), (event.x, event.y)), fill='white', width=40)

        self.xold = event.x
        self.yold = event.y


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Drawing to text Converter")
    root.wm_geometry("%dx%d+%d+%d" % (650, 550, 800, 200))
    root.config(bg='white')
    NumberDetector(root, 10, 10)
    root.mainloop()

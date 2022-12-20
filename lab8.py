import cv2, os
import numpy as np
from PIL import Image, ImageEnhance, ImageDraw, ImageTk
from matplotlib import pyplot as plt
from tkinter import *
from PIL import *




def Face():

    cascadePath = 'haarcascade_frontalface_default.xml'
    faceCascade = cv2.CascadeClassifier(cascadePath)


    recognizer = cv2.face.LBPHFaceRecognizer_create()


    def get_images(path):
        
        image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.happy')]
        
        images = []
        labels = []

        for image_path in image_paths:
            
            gray = Image.open(image_path)
            gray.convert('L')
            image = np.array(gray, 'uint8')
            
            subject_number = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
            
            # Определяем области где есть лица
            faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                images.append(image[y: y + h, x: x + w])
                labels.append(subject_number)
                
                cv2.imshow("", image[y: y + h, x: x + w])
                cv2.waitKey(50)
        return images, labels

        
    # Путь к фотографиям
    path = 'yalefaces'
    
    images, labels = get_images(path)
    cv2.destroyAllWindows()

    # Обучаем программу распознавать лица
    recognizer.train(images, np.array(labels))


    # Создаем список фотографий для распознавания
    image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.happy')]

    for image_path in image_paths:
        # Ищем лица на фотографиях
        gray = Image.open(image_path).convert('L')
        image = np.array(gray, 'uint8')
        faces = faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        
        for (x, y, w, h) in faces:
            
            number_predicted, conf = recognizer.predict(image[y: y + h, x: x + w])


            number_actual = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
            
            if number_actual == number_predicted:
                print ("{} is Correctly Recognized with confidence {}".format(number_actual, conf))
            else:
                print ("{} is Incorrect Recognized as {}".format(number_actual, number_predicted))
            cv2.imshow("Recognizing Face", image[y: y + h, x: x + w])
            cv2.waitKey(1000)

def Menu():
    window = Tk()

    
    window.title("Menu")

    w = window.winfo_screenwidth()
    h = window.winfo_screenheight()
    w = w//2 # середина экрана
    h = h//2 
    w = w - 200 # смещение от середины
    h = h - 200
    window.geometry('300x200+{}+{}'.format(w, h))
    window.configure(bg='#bb85f3')

    btn = Button(window, text="Нахождение счастливых лиц", padx=5, pady=5, command =Face, bg='#eec6ea')  
    btn.pack(anchor="center", padx=50, pady=50)
    


    window.mainloop()

Menu()
from typing import Any

import numpy as np
from keras.models import load_model
from PIL import Image, ImageFilter, ImageTk
from numpy import asarray
import cv2
import matplotlib.pyplot as plt
import imutils
import cnn
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
from tkinter import ttk
from tkinter import filedialog
import tkinter as tk
#                                       Get image data 28 x 28
def imageprepare(image_file: str):
    #image = Image.open(image_file).convert('L')
    image = Image.open(image_file)
    ans_image = Image.new('L', (28, 28), (255))
    #####################     tests
    #img_data = asarray(image)
    #img_data = np.resize(img_data, (28,28))
    #img_pic = Image.fromarray(img_data)
    #img_pic.show()
    #return img_data
    #####################
    
    width = float(image.size[0])
    height = float(image.size[1])
    
    #if width > height:
    nheight = int(round((28.0 / width * height), 0))
    if (nheight == 0):
        nheight = 1     #cause min 1 pixel
    img = image.resize((28, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
    ans_image.paste(img, (0, 0))
    #else:
    #    nwidth = int(round((28.0 / height * width), 0))
    #    if (nwidth == 0):
    #        nwidth = 1      #cause min 1 pixel
    #    img = image.resize((28, nwidth), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
    #    ans_image.paste(img, (0, 0))
    
    #data = asarray(ans_image)
    #ans_image.show()
    return ans_image

def emnist_predict_img(model, img):
    """ Дополнительная подготовка цифр к обработке. """
    #для нового закоменчен, так как уже делал expand
    #img_arr = np.expand_dims(img, axis=0)
    #img_arr = 1 - img_arr/255.0
    #для нового
    img_arr = img
    img_arr[0] = np.rot90(img_arr[0], k=1, axes=(1, 0))
    img_arr[0] = np.fliplr(img_arr[0])
    #plt.imshow(img_arr.reshape(28, 28), cmap=plt.cm.gray)
    #plt.show()
    img_arr = img_arr.reshape((1, 28, 28, 1))
    predict = model.predict([img_arr])
    result = np.argmax(predict, axis=1)
    return chr(cnn.emnist_labels[result[0]])


def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def square_image(img, size_x, size_y, interpolation):
    height, width = img.shape
    #print(f'H: {height}, W: {width}')

    add_x = int((max(height, width) - width) / 2)
    add_y = int((max(height, width) - height) / 2)

    new_img = cv2.copyMakeBorder(img, add_y, add_y, add_x, add_x, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    #plt.imshow(new_img, cmap='gray')
    #plt.show()

    new_img = cv2.resize(new_img, (size_x, size_y), interpolation=interpolation)

    return new_img

def get_letters(img, model):
    letters = ''
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sizeH, sizeW = gray.shape
    #was 127
    ret,thresh1 = cv2.threshold(gray ,120,255,cv2.THRESH_BINARY_INV)
    #thresh1 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #        cv2.THRESH_BINARY_INV,11,9)
    #plt.imshow(thresh1)
    #plt.show()
    dilated = cv2.dilate(thresh1, None, iterations=2)
    #plt.imshow(dilated)
    #plt.show()
    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]
    # loop over the contours
    
    for c in cnts:
        #print(cv2.contourArea(c))
        if cv2.contourArea(c)  < (sizeH * sizeW)* 0.0003:
            continue
        (x, y, w, h) = cv2.boundingRect(c)
        if (w > 1000 or h > 1000):
            continue
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = gray[y:y + h, x:x + w]
        #plt.imshow(roi)
        #plt.show()
        thresh = cv2.threshold(roi, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = square_image(thresh, 28, 28, cv2.INTER_AREA)
        #########
        #plt.imshow(thresh)
        #plt.show()
        ########
        thresh = thresh.astype("float32") / 255.0
        thresh = np.expand_dims(thresh, axis=-1)
        thresh = thresh.reshape(1,28,28,1)
        #ypred = model.predict(thresh)
        #trying roi instead of thresh
        ypred = emnist_predict_img(model, thresh)
        letters += ypred
    return letters, image

def get_word(letter):
    word = "".join(str(letter))
    return word

def open_file():
        filepath = filedialog.askopenfilename()
        if filepath != "":
            picture = Image.open(filepath)
            picture = picture.resize((300, 200), Image.ANTIALIAS)
            pic_forApp = ImageTk.PhotoImage(picture)
            letter, image = get_letters(filepath, netw)
            word = get_word(letter)
            label["text"] = word
            l_pict.configure(image=pic_forApp)
            l_pict.image = pic_forApp

if __name__ == '__main__':
    # обучение модели
    #netw = cnn.emnist_model()
    #cnn.training(netw)
    #netw.save('letters_model.h5')
    netw = load_model('letters_model.h5')
    #letter, image = get_letters('FromM2.jpg', netw)                    #working pretty good
    #letter, image = get_letters('one_2.jpg', netw)
    #letter, image = get_letters('NUMBER0.jpg', netw)
    #letter, image = get_letters('NUMBER0_2.jpg', netw)
    #letter, image = get_letters('NUMBER1.jpg', netw)
    #letter, image = get_letters('NUMBER1_1.jpg', netw)
    #letter, image = get_letters('NUMBER2.jpg', netw)
    #letter, image = get_letters('NUMBER3.jpg', netw)
    #letter, image = get_letters('NUMBER4.jpg', netw)
    #letter, image = get_letters('NUMBER5.jpg', netw)
    #word = get_word(letter)
    #print(word)
    #plt.imshow(image)
    #plt.show()
    # #imageprepare('main.jpg')
    
    #########################
    
    #everything about window
    root = tk.Tk()
    root.title("Phone Recognizer")
    root.geometry("380x310")
    
    #img = PhotoImage(file='C:\\Users\\Ilyas\\AppData\\Local\\Programs\\Python\\Python310\\playground\\phoneRecognizer\\smile.png')
    imgMain = Image.open('C:\\Users\\Ilyas\\AppData\\Local\\Programs\\Python\\Python310\\playground\\phoneRecognizer\\smile.jpg')
    imgMain = imgMain.resize((300, 200), Image.ANTIALIAS)
    #im_file = 'C:\\Users\\Ilyas\\AppData\\Local\\Programs\\Python\\Python310\\playground\\phoneRecognizer\\smile.jpg'
    #imgMain = Image.open(im_file).convert('L')
    imgForApp = ImageTk.PhotoImage(imgMain)
    l_pict = ttk.Label(root, image=imgForApp)
    l_pict.pack(anchor=tk.NW, padx=10, pady=10)
    
    label = ttk.Label(text="Hello! Click the button and select file")
    label.pack(anchor=tk.NW, padx=10, pady=10)            
    
    
    open_button = ttk.Button(text="Open file", command=open_file)
    open_button.pack(anchor=tk.NW, padx=80, pady=10)
    
    root.mainloop()
    
    
    

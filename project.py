from typing import Any

import numpy as np
from keras.models import load_model
from PIL import Image, ImageFilter
from numpy import asarray
import cv2
import matplotlib.pyplot as plt
import imutils
import cnn
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
#                                       Get image data 28 x 28
def imageprepare(image_file: str):
    image = Image.open(image_file).convert('L')
    ans_image = Image.new('L', (28, 28), (255))
    #####################     tests
    img_data = asarray(image)
    #img_data = np.resize(img_data, (28,28))
    img_pic = Image.fromarray(img_data)
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
    
    data = asarray(ans_image)
    ans_image.show()
    return data

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


#def img_to_str(model: Any, image_file: str):
#   """ Перевод распознанных цифр в строку текста. """
#
#    letters, indexes = preprocessing.letters_extract(image_file)
#    s_out = ''
#    for i in range(0, len(letters)):
#        s_out += emnist_predict_img(model, letters[i][2])
#        if (i in indexes):
#            s_out += ' '
#    print('s_out.length = ', len(s_out))
#    return s_out

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

def get_letters(img, model):
    letters = ''
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(gray ,127,255,cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, None, iterations=2)

    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]
    # loop over the contours
    for c in cnts:
        if cv2.contourArea(c) > 10:
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        if (w < 15 or h < 15 or w > 500 or h > 500):
            continue
        l = x
        if (l - 7 > 0):
            l = l - 7
        roi = gray[y:y + h, l:x + w+7]
        #plt.imshow(roi)
        #plt.show()
        thresh = cv2.threshold(roi, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        thresh = cv2.resize(thresh, (28, 28), interpolation = cv2.INTER_CUBIC)
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


if __name__ == '__main__':
    # обучение модели
    netw = cnn.emnist_model()
    #cnn.training(netw)
    #netw.save('letters_model.h5')
    netw = load_model('letters_model.h5')
    #letter, image = get_letters('FromM2.jpg', netw)                    #working pretty good
    #letter, image = get_letters('one_2.jpg', netw)
    #letter, image = get_letters('NUMBER2.jpg', netw)
    #letter, image = get_letters('NUMBER3.jpg', netw)
    #letter, image = get_letters('NUMBER4.jpg', netw)
    letter, image = get_letters('NUMBER5.jpg', netw)
    word = get_word(letter)
    print(word)
    plt.imshow(image)
    plt.show()
    #imageprepare('main.jpg')
    #s_out = emnist_predict_img(netw, imageprepare('fromM.jpg'))            #test just one digits
    #s_out = img_to_str(netw, 'one_1.jpg')                                   #test full number
    #s_out = img_to_str(netw, 'test1_1.jpg')
    #s_out = img_to_str(netw, 'download.jpg')
    #s_out = img_to_str(netw, 'photo_2023-05-24_00-58-54.jpg')
    
    #print(' ')
    #print(s_out)
    #print(' ')
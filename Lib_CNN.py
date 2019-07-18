# -*- coding: utf-8 -*-
"""
Created on Sat July  1 09:55:39 2019

@author: venkatesh avula
"""
import math
import random
from PIL import ImageFont, ImageDraw, Image, ImageFilter
import numpy as np
import keras
from keras.preprocessing import image
import os


#Constants---------------------------------------------------
Fonts = ["Times", "Arial", "Verdana","comic", "Tahoma", "Calibri","Lsans","CALIFR","Rock","ShowG","BRITANIC","BROADW","BRUSHSCI"] #file names : *.ttf fonts
FontsFolder= 'c:/Windows/Fonts/'
ImgSize=32
StddevMax=20#max noise level
FontSize=25 #font size in the image
dirName='Data/'
            
             
characters = ["a","b","c","d","e","f","g","h","i","k",
           "l","m","n","o","p","q","r","s","t","u",
           "v","w","x","y","z", 
           "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", 
           "A", "B", "C", "D","E", "F", "G", "H", "I", "J", "K",
           "L","M", "N", "O","P", "Q", "R", "S", "T", "U",
           "V", "W", "X", "Y","Z"]
nch=len(characters)-1


def add_noise(x, mean, stddev):
    return min(max(0, x+random.normalvariate(mean,stddev)), 255)


def add_noise_one_pixel(img, x, y, mean, stddev):
     r, g, b = img.getpixel((x,y))
     img.putpixel((x,y), (int(add_noise(r, mean, stddev)), int(add_noise(g, mean, stddev)), int(add_noise(b, mean, stddev))))
     return
 

def add_noise_img(img, mean, stddev): 
    X,Y= img.size
    for x in range(X):
        for y in range(Y):
            add_noise_one_pixel(img, x, y, mean, stddev)
    return img      



def MakeDataset2(N):
    font_types=len(Fonts)
    D=ImgSize*ImgSize
    nIndivualFont=N/font_types 
    x=np.zeros((N,D))
    y=np.zeros((N,1))
    
    for i in range(N):
        fonttype=math.floor(i/nIndivualFont)
        data=ImgData(fonttype)
        x[i,:]=data[:,0]
        y[i]=fonttype
    return x,y    
    

def MakeDataset(N):
    font_types=len(Fonts)
    D=ImgSize*ImgSize
    nIndivualFont=N/font_types 
    x=np.zeros((N,D))
    y=np.zeros((N,1),dtype='S4')
    
    for i in range(N):
        fonttype=math.floor(i/nIndivualFont)
        data=ImgData(fonttype)
        x[i,:]=data[:,0]
        y[i]=Fonts[fonttype].encode().decode('utf8')
    return x,y    
       
    
def MakeDataset_CNN(N):
    font_types=len(Fonts)
#    D=ImgSize*ImgSize
    nIndivualFont=N/font_types 
#    x=np.zeros((N,D))
    y=np.zeros((N,1)) #np.zeros((N,1),dtype='S4')
    
    train_image = []
    color_mode = 'grayscale'
    
    for i in range(N):
        fonttype=math.floor(i/nIndivualFont)
        img=ImgSave(fonttype,i)

        img =image.load_img(dirName+str(i)+'.png', color_mode='grayscale',target_size=(ImgSize,ImgSize,1), grayscale=False)  #image.load_img('Data/'+str(i)+'.png', target_size=(ImgSize,ImgSize,1), grayscale=True)
        img = image.img_to_array(img)
        img = img/255
        train_image.append(img)    
        y[i]=fonttype #Fonts[fonttype].encode().decode('utf8')
        
    X = np.array(train_image)     
    return X,y   



def ImgData(fonttype):
    
    D=ImgSize*ImgSize
    vecImage=np.zeros((D,1))
    mean = 0.0
    stddev = random.uniform(0, StddevMax)    # standard deviation
    
    
    img = Image.new('RGB', (ImgSize, ImgSize), color = (255,255,255))

    fontname=Fonts[fonttype]+'.ttf'
        
    fnt = ImageFont.truetype(FontsFolder+fontname, FontSize)
    d = ImageDraw.Draw(img)
    d.text((2,2), "H", font=fnt, fill=(0,0,0)) #test starts at 2 pixels
    #img.save('Times_text_font25T.png')
    
    #noisy font
    img=add_noise_img(img,mean,stddev)
    imgData=list(img.getdata())
    for i in range(D):
        (r,g,b)=imgData[i]
        vecImage[i]=r
        
    return vecImage     


def ImgSave(fonttype,i):
    
#    D=ImgSize*ImgSize
#    vecImage=np.zeros((D,1))
    mean = 0.0
    stddev = random.uniform(0, StddevMax)    # standard deviation
    
    
    img =  Image.new('RGB', (ImgSize, ImgSize), color = (255,255,255))
#    Image.new('L', (ImgSize, ImgSize)) #Image.new('L', (ImgSize, ImgSize), color = (255,255,255))

    fontname=Fonts[fonttype]+'.ttf'
        
    fnt = ImageFont.truetype(FontsFolder+fontname, FontSize)
    d = ImageDraw.Draw(img)
#    d.text((2,2), "H", font=fnt, fill=(0,0,0)) #test starts at 2 pixels
    ich=random.randint(0,nch)
    d.text((2,2), characters[ich], font=fnt, fill=(0,0,0)) 

    #noisy font
    img=add_noise_img(img,mean,stddev)
    if not os.path.exists(dirName):
        os.mkdir(dirName)
        print("Directory " , dirName ,  " Created ")
    
    img.save('Data/'+str(i)+'.png')
        
    return img


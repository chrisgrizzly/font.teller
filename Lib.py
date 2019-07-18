#!/usr/bin/env python
# coding: utf-8

# In[2]:


# -*- coding: utf-8 -*-


#Constants---------------------------------------------------
Fonts = ["arial", "BRITANIC", "BROADW","BRUSHSCI", "calibri", "CALIFR","comic","LSANS","ROCK","SHOWG","tahoma","times","verdana"] #file names : *.ttf fonts
Fonts_abbr = ["Arial", "Britanic", "Broadway", "Brush SMT", "Calibri", "Californian FB", "Comic SMS", "Lucida S", "Rockwell", "Showcard G", "Tahoma", "Times NR", "Verdana"]
FontsFolder= './Fonts/'
ImgSize=32
StddevMax=200#max nose level
FontSize=20 #font size in the image

import math
import random
from PIL import ImageFont, ImageDraw, Image, ImageFilter, ImageChops
import numpy as np
from matplotlib import pyplot as plt
#%matplotlib inline


def add_noise(x, mean, stddev):
    return min(max(0, x+random.normalvariate(mean,stddev)), 255)
    #return min(x, random.normalvariate(mean,stddev))


def add_noise_one_pixel(img, x, y, mean, stddev):
    #r, g, b = img.getpixel((x,y))
    #img.putpixel((x,y), (int(add_noise(r, mean, stddev)), int(add_noise(g, mean, stddev)), int(add_noise(b, mean, stddev))))
    r = img.getpixel((x,y))
    img.putpixel((x,y), int(add_noise(r, mean, stddev)))
    return
 

def add_noise_img(img, mean, std_min, std_max): 
    X,Y= img.size
    stddev = np.random.uniform(std_min, std_max)
    #print(stddev)
    for x in range(X):
        for y in range(Y):
            add_noise_one_pixel(img, x, y, mean, stddev)
    return img      

def deform_img(img, strech_std, offset_std, rotation_std):
    #Center the image
    X_pixels = []
    Y_pixels = []
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            r = img.getpixel((x, y))
            if r != 0:
                X_pixels.append(x)
                Y_pixels.append(y)
    X_max = max(X_pixels)
    X_min = min(X_pixels)
    Y_max = max(Y_pixels)
    Y_min = min(Y_pixels)
    width = X_max - X_min + 1
    height = Y_max - Y_min + 1
    offset_x = int(round((ImgSize-X_max-X_min)/2))
    offset_y = int(round((ImgSize-Y_max-Y_min)/2))
    img = ImageChops.offset(img, offset_x, offset_y)
    
    #Rotate the image
    img = img.rotate(min(np.random.normal(0.0, rotation_std),15))
    
    #Corner the image
    img = ImageChops.offset(img, -X_min, -Y_min)
    #print(img.size)
    X_pixels = []
    Y_pixels = []
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            r = img.getpixel((x, y))
            if r != 0:
                X_pixels.append(x)
                Y_pixels.append(y)
    X_max = max(X_pixels)
    X_min = min(X_pixels)
    Y_max = max(Y_pixels)
    Y_min = min(Y_pixels)
    width = X_max - X_min + 1
    height = Y_max - Y_min + 1
    
    #stretch the image
    max_strech_width = np.floor(ImgSize**2/width).astype(int)
    max_strech_height = np.floor(ImgSize**2/height).astype(int)
    strech_width = min(np.floor((ImgSize*(1+abs(np.random.normal(0.0, strech_std))))).astype(int), max_strech_width)
    strech_height = min(np.floor((ImgSize*(1+abs(np.random.normal(0.0, strech_std))))).astype(int), max_strech_height)
    #print(strech_width, strech_height)
    #background = Image.new('L', (ImgSize, ImgSize), color = 255)
    img = img.resize((strech_width, strech_height))
    img = img.crop((0,0,ImgSize,ImgSize))
    #background.paste(img, (0, 0))
    #img = background
    
    #Center the image
    X_pixels = []
    Y_pixels = []
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            r = img.getpixel((x, y))
            if r != 0:
                X_pixels.append(x)
                Y_pixels.append(y)
    X_max = max(X_pixels)
    X_min = min(X_pixels)
    Y_max = max(Y_pixels)
    Y_min = min(Y_pixels)
    width = X_max - X_min + 1
    height = Y_max - Y_min + 1
    offset_x = int(round((ImgSize-X_max-X_min)/2))
    offset_y = int(round((ImgSize-Y_max-Y_min)/2))
    img = ImageChops.offset(img, offset_x, offset_y)
    
    #Shake the image around enter
    X_pixels = []
    Y_pixels = []
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            r = img.getpixel((x, y))
            if r != 0:
                X_pixels.append(x)
                Y_pixels.append(y)
    X_max = max(X_pixels)
    X_min = min(X_pixels)
    Y_max = max(Y_pixels)
    Y_min = min(Y_pixels)
    offset_x = int(math.floor(max(min(np.random.normal(0.0, offset_std),ImgSize-X_max),-X_min)))
    offset_y = int(math.floor(max(min(np.random.normal(0.0, offset_std),ImgSize-Y_max),-Y_min)))
    img = ImageChops.offset(img, offset_x, offset_y)
    
    return img
    

def MakeDataset(N, std_min, std_max, mean, strech_std, offset_std, rotation_std):
    font_types=len(Fonts)
    #print(font_types)
    D=ImgSize*ImgSize
    nIndivualFont=N/font_types 
    #print(nIndivualFont)
    x=np.zeros((N,D))
    y=np.zeros((N,1), dtype = int)
    
    for i in range(N):
        fonttype=int(math.floor(i/nIndivualFont))
        data=ImgData(fonttype, std_min, std_max, mean, strech_std, offset_std, rotation_std)
        x[i,:]=data[:,0]
        y[i]=int(fonttype)#.encode().decode('utf8')
    return x,y    

def MakeDatasetCycle(N, std_min, std_max, mean, strech_std, offset_std, rotation_std):
    font_types=len(Fonts)
    #print(font_types)
    D=ImgSize*ImgSize
    #nIndivualFont=N/font_types 
    #print(nIndivualFont)
    x=np.zeros((N,D))
    y=np.zeros((N,1), dtype = int)
    
    for i in range(N):
        fonttype=int(round(i%font_types))
        #print(fonttype)
        data=ImgData(fonttype, std_min, std_max, mean, strech_std, offset_std, rotation_std)
        x[i,:]=data[:,0]
        y[i]=int(fonttype)#.encode().decode('utf8')
    return x,y   
       
    

def ImgData(fonttype, std_min, std_max, mean, strech_std, offset_std, rotation_std):
    
    D=ImgSize*ImgSize
    vecImage=np.zeros((D,1))
    #mean = 0.0
    #stddev = random.uniform(0, StddevMax)    # standard deviation
    offset_x = 0
    offset_y = 10
    
    
    #img = Image.new('RGB', (ImgSize, ImgSize), color = (255,255,255))
    img = Image.new('L', (ImgSize, ImgSize), color = 0)

    fontname=Fonts[fonttype]+'.ttf'
        
    fnt = ImageFont.truetype(FontsFolder+fontname, FontSize)
    d = ImageDraw.Draw(img)
    #d.text((0,0), "H", font=fnt, fill=(0,0,0))
    d.text((0,0), "H", font=fnt, fill=255)
    #img.save('Times_text_font25T.png')
    
    #noisy font
    img = deform_img(img, strech_std, offset_std, rotation_std)
    img = add_noise_img(img,mean,std_min, std_max)
    imgData = list(img.getdata())
    for i in range(D):
        #(r,g,b)=imgData[i]
        r = imgData[i]
        vecImage[i] = r
        
    return vecImage     


def ShowDataImage(dataset):
    x,y = dataset
    #print(x.shape)
    #test=np.zeros((4,1024)).reshape((4,32,32))
    #np.split(x[0], 32, axis=0)
    #print(test.shape)
    #print(x)
    x = x.reshape((x.shape[0],ImgSize,ImgSize))
    fig = plt.figure(figsize=(20, np.ceil(x.shape[0]/20)*1.5))
    for k in range(x.shape[0]):
        ax = fig.add_subplot(np.ceil(x.shape[0]/20), 20, k+1, xticks=[], yticks=[])
        #print(x[k].reshape((32,32)))
        ax.imshow(x[k], cmap=plt.cm.bone)
        #ax.set_title(f"{k+1}")
        #ax.set_xlabel(f"{Fonts_abbr[int(y[k])]}")
    return

def ShowDataImageCycle(dataset):
    x,y = dataset
    #print(x.shape)
    #test=np.zeros((4,1024)).reshape((4,32,32))
    #np.split(x[0], 32, axis=0)
    #print(test.shape)
    #print(x)
    x = x.reshape((x.shape[0],ImgSize,ImgSize))
    fig = plt.figure(figsize=(26, np.ceil(x.shape[0]/26)))
    for k in range(x.shape[0]):
        ax = fig.add_subplot(np.ceil(x.shape[0]/26), 26, k+1, xticks=[], yticks=[])
        #print(x[k].reshape((32,32)))
        ax.imshow(x[k], cmap=plt.cm.bone)
        #ax.set_title(f"{k+1}")
        #ax.set_xlabel(f"{Fonts_abbr[int(y[k])]}")
    plt.subplots_adjust(wspace=0, hspace=0)
    return

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import json
import sys
from tensorflow import keras


def find_focusbox(file):
    img = cv2.imread(file)
    
    
    lower = np.array([230],dtype = "uint8")
    upper = np.array([255],dtype = "uint8")
    
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, lower, upper)
    gray = cv2.bitwise_and(gray,gray,mask = mask)
    #####################HOUGH LINES##########################################
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    #print(edges)
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength=50,maxLineGap=10)
    
    img = cv2.imread(file)
    if len(lines) == 2:
        pass
        #print("x: ",lines[0][0][0],"y: ",min(lines[1][0][1],lines[0][0][1]),"w: ",lines[0][0][2]-lines[0][0][0],"h: ",abs(lines[1][0][3]-lines[0][0][3]))
    cv2.rectangle(img,(lines[0][0][0],lines[0][0][1]),(lines[0][0][2],lines[1][0][1]),(0,255,0),2)
    cv2.imwrite("new/"+file,img)
    
    return [lines[0][0][0],min(lines[1][0][1],lines[0][0][1]),lines[0][0][2]-lines[0][0][0],abs(lines[1][0][3]-lines[0][0][3])]



def give_flags(file):
    img = cv2.imread(file)
    #print(file)
    flags = [0,0,0,0,0,0]
    ind = 0
    for template_file in os.listdir('ref'):
            
        template = cv2.imread('ref/'+template_file,0)
        w, h = template.shape[::-1]
        
        gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        res_img = cv2.matchTemplate(gray_img,template,cv2.TM_CCOEFF_NORMED)
        
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_img)
        # top_left = max_loc
        # bottom_right = (top_left[0] + w, top_left[1] + h)
        # cv2.rectangle(img,top_left, bottom_right, (0,255,0), 2)
        
        
        threshold = 0.95
        loc = np.where( res_img >= threshold)
        
        if loc[0].size > 0:
          flags[ind] = 1
        ind = ind+1
        
    return flags


def return_roi(file,template_file):
    img = cv2.imread(file)
    template = cv2.imread(template_file,0)
    w, h = template.shape[::-1]
    
    gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    res_img = cv2.matchTemplate(gray_img,template,cv2.TM_CCOEFF_NORMED)
    
    threshold = 0.95
    loc = np.where( res_img >= threshold)
    
    if loc[0].size > 0:
        #print(loc[0][0],loc[1][0])
        #print(template_file)
        for pt in zip(*loc[::-1]):
            cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2)
            x,y,wf,hf = find_focusbox(file)
            return [pt[0]-x,pt[1]-y,w-wf,h-hf]
        
        
        
model2 = keras.models.load_model('Notification-app-final')
states = [i[:-4] for i in os.listdir('notification-screenshots')]
icons_path = ['ref/'+i for i in os.listdir('ref')]
for i in os.listdir('notification-screenshots'):    
    file_path = 'notification-screenshots/' + i
    focus_box = find_focusbox(file_path)
    flags = give_flags(file_path)
    
    if sum(flags) != 0:
        focus_box = return_roi(file_path,icons_path[flags.index(1)])
        
    input_values = np.array(focus_box+flags,dtype="float")
    input_values = input_values.reshape((1,10))
    #print(input_values)

    input_values[:,[0,1,2,3]] = np.log(abs(input_values[:,[0,1,2,3]]))/6.0
    #print(input_values)
    print("Input image for hybrid model : ", i)
    print("Predicted State ",states[np.argmax(model2.predict(input_values))])
    
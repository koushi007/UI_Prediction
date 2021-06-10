import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import json


def make_focus_box(li):
    fb = [("x",int(li[0])),("y",int(li[1])),("w",int(li[2])),("h",int(li[3]))]
    return dict(fb)


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
        print("x: ",lines[0][0][0],"y: ",min(lines[1][0][1],lines[0][0][1]),"w: ",lines[0][0][2]-lines[0][0][0],"h: ",abs(lines[1][0][3]-lines[0][0][3]))
    #cv2.rectangle(img,(lines[0][0][0],lines[0][0][1]),(lines[0][0][2],lines[1][0][1]),(0,255,0),2)
    #cv2.imwrite("new/"+file,img)
    
    return [lines[0][0][0],min(lines[1][0][1],lines[0][0][1]),lines[0][0][2]-lines[0][0][0],abs(lines[1][0][3]-lines[0][0][3])]
    
    

def temp_match(file):
    
    img = cv2.imread('notification-screenshots/'+file)
    
    print(file)
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
            print(loc[0][0],loc[1][0])
            print(template_file)
            for pt in zip(*loc[::-1]):
                cv2.rectangle(img, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2)
                x,y,wf,hf = focus_box[file[:-4]]
                return [pt[0]-x,pt[1]-y,w-wf,h-hf,template_file]
            
            
            plt.imshow(img)
            
    return [-1,-1,-1,-1]
            

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
import json
# IMREAD_COLOR loads the image in the BGR 8-bit format. This is the default that is used here.
# IMREAD_UNCHANGED loads the image as is (including the alpha channel if present)
# IMREAD_GRAYSCALE loads the image as an intensity one
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
print(cv2.__version__)

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



states = [i[:-4] for i in os.listdir('notification-screenshots')]
icons_path = ['ref/'+i for i in os.listdir('ref')]

for file in os.listdir('notification-screenshots'):
     
    img = cv2.imread('notification-screenshots/'+file)
    flags = give_flags('notification-screenshots/'+file)
    
    
    
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    ret,thresh = cv2.threshold(gray,15,255,2) 
     
    contours,h = cv2.findContours(thresh,3,2) 
    outer_contour = []
    for cnt in contours: 
         
        (x,y,w,h) = cv2.boundingRect(cnt)
        if w > 100 and h>50 and y < 150 and w < 300 :
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            outer_contour = [x,y,w,h]
    
    
    ret,thresh = cv2.threshold(gray,15,255,1) 
     
    contours,h = cv2.findContours(thresh,3,2) 
     
    for cnt in contours: 
        (x,y,w,h) = cv2.boundingRect(cnt)
        if w > 100 and h>50 and y < 150 and w < 300 :
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            outer_contour = [x,y,w,h]

    
    ret,thresh = cv2.threshold(gray,230,255,1) 
     
    contours,h = cv2.findContours(thresh,3,2) 
    focus_box = []
    for cnt in contours: 
        (x,y,w,h) = cv2.boundingRect(cnt)
        if w > 70 and w < 200:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            focus_box = [x,y,w,h]
    cv2.imwrite('new/'+file,img)
    
    x1,y1,x2,y2 = focus_box[0],focus_box[1],focus_box[0]+focus_box[2],focus_box[1]+focus_box[3]
    cropped_img = img[y1:y2,x1:x2]
    cv2.imwrite("buttons/"+file,cropped_img)
    text = pytesseract.image_to_string(cropped_img)
    text = text.encode("ascii","ignore")
    text = text.decode().lstrip().rstrip()
    #print(text)
    
            
    outer_contour_base = outer_contour[1]+outer_contour[3]
    focus_box_base = focus_box[1]+focus_box[3]
    
    relative_diff = (outer_contour_base-focus_box_base)*(1.0/focus_box[3])
    print("Input Image given: ",file)
    predicted_state = ""
    if flags[0]:
        if relative_diff < 0.5:
            predicted_state = "cam_move#main#close"
        elif relative_diff < 1.5:
            predicted_state = "cam_move#main#fullscreen"
    
    if flags[1]:
        if relative_diff < 0.5:
            predicted_state = "network#main#connected"
            
    if flags[2]:
        if relative_diff < 0.5:
            predicted_state = "network#main#disconnected"
            
    if flags[3]:
        if relative_diff < 0.5:
            predicted_state = "sh#alarm#settings"
        elif relative_diff < 1.5:
            predicted_state = "sh#alarm#skip_today"
        elif relative_diff < 2.5:
            predicted_state = "sh#alarm#startnow"
            
    
    if flags[4]:
        if text == 'Cancel' :
            predicted_state = "st_pin#main#cancel"
        else:
            predicted_state = "st_timeout#main#ok"
        
    if flags[5]:
        if relative_diff < 0.5 and text == 'Cancel' :
            predicted_state = "usb_in#main#cancel"
        elif relative_diff<0.5:
            predicted_state = "usb_out#exit#ok"
        elif relative_diff < 1.5:
            predicted_state = "usb_in#main#browse"
        
    if sum(flags) == 0:
        if relative_diff < 0.5:
            predicted_state = "appinstall#main#close"
        elif relative_diff < 1.5:
            predicted_state = "appinstall#main#open"
        
    
    print("predicted state:",predicted_state)
    print("\n")
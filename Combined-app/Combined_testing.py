import cv2
import os
import numpy as np
import pandas as pd
from pandas.core.common import flatten
from numpy import argmax
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from keras.utils import np_utils
from keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow import keras

ref = 'org.tizen.netflix-app'
data = []

df_iconstates = pd.read_csv('Netflix_iconstates.csv')
list_iconstates = list(df_iconstates['Icon_state'].values)
dict_iconstates = dict(list(df_iconstates.values))


def find_icon(gray_img,start):
    for template_file in os.listdir(ref):
        if not template_file.endswith('.png') or not template_file.startswith(start):
            continue
            
        template = cv2.imread(ref+'/'+template_file,0)
        w, h = template.shape[::-1]
        wi, hi = gray_img.shape[::-1]
        
        if w > wi or h > hi:
            continue
        
        res_img = cv2.matchTemplate(gray_img,template,cv2.TM_CCOEFF_NORMED)
        
        
        threshold = 0.8
        loc = np.where( res_img >= threshold)
        
        if loc[0].size > 0:
            icon_state = template_file[:-4]
            if icon_state[-1] == '2' and icon_state[:4] != "home":
                return icon_state[:-1]
            return icon_state
        
    return ''
    


for file in os.listdir('Netflix-Testing'):
    
    data_row_green = []
    data_row_blue = []
    data_row_icons = [0 for i in range(25)]
    data_row = []
    
     
    #img = cv2.imread('netflix-screenshots/home#main#thumbnail-42.png')
    img = cv2.imread('Netflix-Testing/'+file)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    #blurred = cv2.GaussianBlur(gray, (1, 1), 0)

    ##################### Menu detection and content detection ########################
    ret,thresh = cv2.threshold(gray,230,255,0)     #27 is default
    #thresh = cv2.GaussianBlur(thresh, (7, 7), 0)

    contours,hierarchy = cv2.findContours(thresh,3,cv2.CHAIN_APPROX_SIMPLE) 
    lis = [i[0][0][0] == 0 for i in contours]
    ind = - 1
    try:
        ind = lis.index(True)
    except:
        pass
    box = []            
                    
    for i in range(len(contours)):
        cnt = contours[i]
        
        (x,y,w,h) = cv2.boundingRect(cnt)
        if y>200 and w>50 and h<40 and (hierarchy[0][i][3] == ind or hierarchy[0][i][3] == -1):
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            box = [x,y,w,h]
            
    #print(file,box)
    icon_state = ''
    if box == []:
        icon_state = find_icon(thresh[100:470,0:80],'menu')
    else:
        icon_state = find_icon(gray[box[1]:box[1]+box[3],box[0]:box[0]+box[2]],'content')
    
    # ###################### Highlighted Rectangle blue ########################
    ret,thresh = cv2.threshold(gray,230,255,0)     #27 is default
     
    contours,hierarchy = cv2.findContours(thresh,3,cv2.CHAIN_APPROX_SIMPLE) 
    lis = [i[0][0][0] == 0 for i in contours]
    ind = -1
    try:
        ind = lis.index(True)
    except:
        pass
    outer_contour = []
    for i in range(len(contours)):
        cnt = contours[i]
        approx = cv2.approxPolyDP(cnt, 0.02* cv2.arcLength(cnt, True), True)
        if len(approx) == 4 :
            
            (x,y,w,h) = cv2.boundingRect(approx)
            if w>15 and h>15 and (hierarchy[0][i][3] == ind or hierarchy[0][i][3]==-1):
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                outer_contour = [x,y,w,h]
                data_row_blue.append(outer_contour)

                if icon_state == '' and y>5 and x>60:
                    icon_state = find_icon(gray[y-5:y+h+5,x-60:x+5],'home')

        
    # print(icon_state)        
    # print("###########################")

    ########################## Boxes with 0 pixesl padding green ##############################
    ret,thresh = cv2.threshold(gray,0,255,1)     #27 is default
    #thresh = cv2.GaussianBlur(thresh, (7, 7), 0)

    contours,hierarchy = cv2.findContours(thresh,3,cv2.CHAIN_APPROX_SIMPLE) 
    lis = [i[0][0][0] == 0 for i in contours]
    ind = lis.index(True)
    outer_contour = []            
                    
    for i in range(len(contours)):
        cnt = contours[i]
        approx = cv2.approxPolyDP(cnt, 0.02* cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            
            (x,y,w,h) = cv2.boundingRect(cnt)
            # if w>50 and h>50 and (hierarchy[0][i][3] == ind or hierarchy[0][i][3] == -1):
            if (w> 15 and h > 15 and w<200) and (hierarchy[0][i][3] == ind or hierarchy[0][i][3] == -1):
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                outer_contour = [x,y,w,h]
                data_row_green.append(outer_contour)
                
                
    data_row_green.sort()
    data_row_blue.sort()
    data_row = list(flatten(data_row_blue)) + list(flatten(data_row_green))
    
    
    sz = len(data_row)
    for i in range(60-int(sz/4)):
        data_row = data_row + [0,0,720,576]
    try:
        ind = dict_iconstates[icon_state]
        data_row_icons[ind] = 1
    except:
        pass
    data_row = data_row + data_row_icons
    data_row.append(1)
    data.append(data_row)
    

    # cv2.imwrite('new-nt/'+file,img)
    # thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
    # cv2.imwrite('new-nt/thresh'+file,thresh)

for file in os.listdir('youtube_Testing'):
    
    data_row_green = []
    data_row_red = []
    data_row_blue = []
    data_row_icons = [0 for i in range(25)]
    data_row = []
    
     
    #img = cv2.imread('netflix-screenshots/home#main#thumbnail-42.png')
    img = cv2.imread('youtube_Testing/'+file)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    icon_state = ''

    ########################## All Focus Boxes ##############################
    ret,thresh = cv2.threshold(blurred,27,255,1)     #27 is default
     
    contours,hierarchy = cv2.findContours(thresh,3,cv2.CHAIN_APPROX_SIMPLE) 
    lis = [i[0][0][0] == 0 for i in contours]
    ind = lis.index(True)
    outer_contour = []
    thumbnail_present = False
    thumb_first_coords = [576,720]
    
    for i in range(len(contours)):
        cnt = contours[i]
        approx = cv2.approxPolyDP(cnt, 0.07* cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            
            (x,y,w,h) = cv2.boundingRect(approx)
            if w>50 and h>50 and (hierarchy[0][i][3] == ind or hierarchy[0][i][3] == -1) and x > 0:
                thumbnail_present = True
                if x < thumb_first_coords[0] and x > 0:
                    thumb_first_coords = [x,y]
                
                    
    for i in range(len(contours)):
        cnt = contours[i]
        approx = cv2.approxPolyDP(cnt, 0.07* cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            
            (x,y,w,h) = cv2.boundingRect(approx)
            if w>6 and h>6  and (hierarchy[0][i][3] == ind or hierarchy[0][i][3] == -1):
                if thumbnail_present and (w < 50 or h < 50) and x >= thumb_first_coords[0]-5:
                    continue
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                outer_contour = [x,y,w,h]
                data_row_green.append(outer_contour)

    #####################Highlighted thumbnail########################
    ret,thresh = cv2.threshold(gray,230,255,1)     #27 is default
     
    contours,hierarchy = cv2.findContours(thresh,3,cv2.CHAIN_APPROX_SIMPLE) 
    lis = [i[0][0][0] == 0 for i in contours]
    ind = lis.index(True)
    largest_contour = [0,0,-1,-1]
    for i in range(len(contours)):
        cnt = contours[i]
        approx = cv2.approxPolyDP(cnt, 0.07* cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            
            (x,y,w,h) = cv2.boundingRect(approx)
            if w>largest_contour[2] and h>largest_contour[3] and (hierarchy[0][i][3] == ind or hierarchy[0][i][3]==-1):
                if x != 0 and y!=0:
                    largest_contour = [x,y,w,h]
    [x,y,w,h] = largest_contour
    if w > 50 and h > 50 :
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
        outer_contour = [x,y,w,h]
        data_row_red.append(outer_contour)

    ######################3Highlighted menu item########################
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    ret,thresh = cv2.threshold(blurred,230,255,1)     #27 is default
     
    contours,hierarchy = cv2.findContours(thresh,3,cv2.CHAIN_APPROX_SIMPLE) 
    lis = [i[0][0][0] == 0 for i in contours]
    ind = lis.index(True)
    outer_contour = []
    for i in range(len(contours)):
        cnt = contours[i]
        approx = cv2.approxPolyDP(cnt, 0.07* cv2.arcLength(cnt, True), True)
        if len(approx) >= 4 and len(approx) <= 8:
            
            (x,y,w,h) = cv2.boundingRect(approx)
            if w>5 and h>5 and (hierarchy[0][i][3] == ind or hierarchy[0][i][3]==-1):
                if thumbnail_present and x >= thumb_first_coords[0]-5:
                    continue
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                outer_contour = [x,y,w,h]
                data_row_blue.append(outer_contour)
                
    #             if icon_state == '' and x>0 and w>10 and h>10 and  w<20 and h<20:
    #                 icon_state = find_icon(gray[y:y+h,x:x+w],'')
    #                 print(x,y,w,h)

        
    # print(icon_state)        
    # print("###########################")

                
    data_row_green.sort()
    data_row_blue.sort()
    data_row_red.sort()
    
    data_row = list(flatten(data_row_red)) + list(flatten(data_row_blue)) + list(flatten(data_row_green))
    
    
    

    
    sz = len(data_row)
    for i in range(60-int(sz/4)):
        data_row = data_row + [0,0,720,576]
    data_row = data_row + data_row_icons
    data_row.append(0)
    #data_row.append(data_row_icons)
    #data_row.append(file.split('-')[0])
    data.append(data_row)
    

    
    
def col_names(i):
    return ["x{}".format(i),"y{}".format(i),"w{}".format(i),"h{}".format(i)]

columns = []
for i in range(60):
    columns = columns + col_names(i+1)
columns = columns + list_iconstates

columns.append('APP')
df = pd.DataFrame(data,columns = columns)

X = df.values[:,:]
# ensure all data are floating point values
X = X.astype('float32')
X[:,:240] = np.log(X[:,:240]+4)/6.0

model = keras.models.load_model('Combined-toy0.976')

df_statesmap = pd.read_csv('Combined_states_map.csv')
dict_statesmap = dict(list(df_statesmap.values))

input_states = [i.split('-')[0] for i in os.listdir('Netflix-Testing')] + [i.split('-')[0] for i in os.listdir('youtube_Testing')]
predicted_states = []
for i in range(X.shape[0]):
    print(dict_statesmap[argmax(model.predict(X)[i])])
    predicted_states.append(dict_statesmap[argmax(model.predict(X)[i])])
result = [1 if predicted_states[i] == input_states[i] else 0 for i in range(X.shape[0])]

print("Test accuracy : " + str(np.sum(result)) + " images predicted correctly out of "+str(len(input_states))) 
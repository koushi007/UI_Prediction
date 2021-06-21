import cv2
import os
import pandas as pd
from pandas.core.common import flatten
data = []


def inc(var,i,maxi):
    if var+i > maxi:
        return maxi
    else:
        return var + i

def dec(var,i,mini):
    if var-i < mini:
        return mini
    else:
        return var - i

def generate_data(data_row):
    data_gen = [data_row]
    for j in range(1,4):
        temp = [ inc(data_row[i],j,720) if i%4==0 else data_row[i] for i in range(len(data_row)-1)]
        temp.append(data_row[len(data_row)-1])
        data_gen.append(temp)
    
        temp = [inc(data_row[i],j,576) if i%4==1  else data_row[i] for i in range(len(data_row)-1)]
        temp.append(data_row[len(data_row)-1])
        data_gen.append(temp)
    
        temp = [inc(data_row[i],j,720-data_row[i-2]) if i%4==2  else data_row[i] for i in range(len(data_row)-1)]
        temp.append(data_row[len(data_row)-1])
        data_gen.append(temp)

        temp = [inc(data_row[i],j,576-data_row[i-2]) if i%4==3   else data_row[i] for i in range(len(data_row)-1)]
        temp.append(data_row[len(data_row)-1])
        data_gen.append(temp)
        
        temp = [dec(data_row[i],j,0) if i%4==0 else data_row[i] for i in range(len(data_row)-1)]
        temp.append(data_row[len(data_row)-1])
        data_gen.append(temp)
    
        temp = [dec(data_row[i],j,0) if i%4==1 else data_row[i]  for i in range(len(data_row)-1)]
        temp.append(data_row[len(data_row)-1])
        data_gen.append(temp)
    
        temp = [dec(data_row[i],j,5) if i%4==2  else data_row[i] for i in range(len(data_row)-1)]
        temp.append(data_row[len(data_row)-1])
        data_gen.append(temp)

        temp = [dec(data_row[i],j,5) if i%4==3 else data_row[i]  for i in range(len(data_row)-1)]
        temp.append(data_row[len(data_row)-1])
        data_gen.append(temp)
        
    return data_gen


for file in os.listdir('youtube-screenshots'):
    data_row_green = []
    data_row_red = []
    data_row_blue = []
    data_row = []
    
     
    #img = cv2.imread('youtube-screenshots/home#grid#thumbnail2-35.png')
    img = cv2.imread('youtube-screenshots/'+file)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

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
                
    data_row_green.sort()
    data_row_blue.sort()
    data_row_red.sort()
    
    data_row = list(flatten(data_row_red)) + list(flatten(data_row_blue)) + list(flatten(data_row_green))
    
    
    sz = len(data_row)
    for i in range(60-int(sz/4)):
        data_row = data_row + [0,0,720,576]
    data_row.append(file.split('-')[0])
    data = data+generate_data(data_row)
    cv2.imwrite('new-yt/'+file,img)
    thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
    cv2.imwrite('new-yt/thresh'+file,thresh)


def col_names(i):
    return ["x{}".format(i),"y{}".format(i),"w{}".format(i),"h{}".format(i)]

columns = []
for i in range(60):
    columns = columns + col_names(i+1)
    
columns.append('State')
df = pd.DataFrame(data,columns = columns)

df.to_csv('youtube_train_ro.csv')
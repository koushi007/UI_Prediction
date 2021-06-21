import cv2
import os

for file in os.listdir('youtube-screenshots'):
     
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
            if w>6 and h>6 and (hierarchy[0][i][3] == ind or hierarchy[0][i][3] == -1):
                if thumbnail_present and (w < 50 or h < 50) and x >= thumb_first_coords[0]-5:
                    continue
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                outer_contour = [x,y,w,h]

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
    if w > 50 and h > 50:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,0,255), 2)
        outer_contour = [x,y,w,h]

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

    # ret,thresh = cv2.threshold(gray,15,255,1) 
     
    # contours,h = cv2.findContours(thresh,3,2) 
     
    # for cnt in contours: 
    #     (x,y,w,h) = cv2.boundingRect(cnt)
    #     if w > 100 and h>50 and y < 150 and w < 300 :
    #         cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    #         outer_contour = [x,y,w,h]

    
    # ret,thresh = cv2.threshold(gray,230,255,1) 
     
    # contours,h = cv2.findContours(thresh,3,2) 
    # focus_box = []
    # for cnt in contours: 
    #     (x,y,w,h) = cv2.boundingRect(cnt)
    #     if w > 70 and w < 200:
    #         cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
    #         focus_box = [x,y,w,h]
    cv2.imwrite('new-yt/'+file,img)
    thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
    cv2.imwrite('new-yt/thresh'+file,thresh)

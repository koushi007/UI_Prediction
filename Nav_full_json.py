import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pytesseract
import json
from utils import *
# IMREAD_COLOR loads the image in the BGR 8-bit format. This is the default that is used here.
# IMREAD_UNCHANGED loads the image as is (including the alpha channel if present)
# IMREAD_GRAYSCALE loads the image as an intensity one
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe'
# pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'  # tesseract path for Ubuntu server

print(cv2.__version__)

focus_box = dict()
focus_roi_diff = dict()
st_with_noimage = dict()
file_ordering = []
focus_box_list = []


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
            
            
            
main_json = dict()
main_json["meta_version"] = 0.1
main_json["app_name"] = "com.samsung.tv.notifications-app"
main_json["algorithms"] = []
main_json["state_descriptions"] = []


for file in os.listdir('notification-screenshots'):
    print(file)
    file_ordering.append(file)
    focus_box[file[:-4]] = find_focusbox('notification-screenshots/'+file)
    focus_box_list.append(focus_box[file[:-4]])
    focus_roi_diff[file[:-4]] = temp_match(file)
    if len(focus_roi_diff[file[:-4]]) == 4:
        st_with_noimage[file[:-4]] = True
    else:
        st_with_noimage[file[:-4]] = False
        
    #######################Reading the button#################################
    img = cv2.imread('notification-screenshots/'+file)
    x1,y1 = focus_box[file[:-4]][0],focus_box[file[:-4]][1],
    x2,y2 = focus_box[file[:-4]][0]+focus_box[file[:-4]][2],focus_box[file[:-4]][1]+focus_box[file[:-4]][3]
    cropped_img = img[y1:y2,x1:x2]
	
	# On ubuntu server pytesseract.image_to_string() crashes with png data, so storing the data in jpg
    cropped_file = "buttons/"+file[:-3] + "jpg"
    print(cropped_file)
    cv2.imwrite(cropped_file,cropped_img)
    text = pytesseract.image_to_string(Image.open(cropped_file))
    text = text.encode("ascii","ignore")
    text = text.decode().lstrip().rstrip()
    print(text)
    
    st_desc = dict()
    st_desc["state"] = file[:-4]
    st_desc["type"] = "button"
    st_desc["button_title"] = {"en" : [text]}
    
    main_json["state_descriptions"].append(st_desc)




from scipy.cluster.hierarchy import dendrogram, linkage

linked = linkage(focus_box_list,'centroid')

#labelList = range(1, 11)

# Following looks useful for debugging only. Please enable it only on Windows. On ubuntu server it crashes
#plt.figure(figsize=(10, 7))
#dendrogram(linked)
#plt.show()

num = np.size(np.where(linked[:,2] < 5))

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=num, affinity='euclidean', linkage='ward')
grouping = cluster.fit_predict(focus_box_list)

group_list = [i for i in range(num)]

for i in range(0,num):
    print("Group #{}".format(i))
    dum = []
    for g in range(0,len(file_ordering)):
        if grouping[g] == i:
            print(file_ordering[g])
            dum.append(file_ordering[g][:-4])
    group_list[i] = dum
    
group_list.sort(key = lambda x : len(x))


algo = dict()
algo["algo_name"] = "color_filtered_boundingboxes"
algo["min_color"] = [230,230,230]
algo["max_color"] = [255,255,255]
main_json["algorithms"].append(algo)

final_images = []

for st_list in group_list:
    if len(st_list) == 1:
        if st_with_noimage[st_list[0]]:
            algo = dict()
            algo["algo_name"] = "search_focusbox"
            parem_dict = dict()
            parem_dict["state"] = st_list[0]
            parem_dict["comment"] = st_list[0]
            parem_dict["focus_box"] = make_focus_box(focus_box[st_list[0]])
            algo["parameters"] = [parem_dict]
            main_json["algorithms"].append(algo)
        else:
            algo = dict()
            algo["algo_name"] = "template_match"
            algo["comment"] = st_list[0]
            algo["match_threshold"] = 0
            algo["focus_box"] = make_focus_box(focus_box[st_list[0]])
            algo["focus_box_margin"] = make_focus_box([0,0,0,0])
            algo["focus_roi_diff"] = make_focus_box(focus_roi_diff[st_list[0]][:4])
            parem_dict = dict()
            parem_dict["state"] = st_list[0]
            parem_dict["asset"] = focus_roi_diff[st_list[0]][4]
            algo["parameters"] = [parem_dict]
            
            main_json["algorithms"].append(algo)
            
            
    else:
        algo = dict()
        algo["algo_name"] = "template_match"
        algo["match_threshold"] = 0
        algo["comment"] = " "
        algo["parameters"] = []
        for st in st_list:
            if st_with_noimage[st]:
                final_images.append(st)
                continue
            
            algo["comment"] += " " + st
            algo["focus_box"] = make_focus_box(focus_box[st])
            algo["focus_box_margin"] = make_focus_box([0,0,0,0])
            algo["focus_roi_diff"] = make_focus_box(focus_roi_diff[st][:4])
            parem_dict = dict()
            parem_dict["state"] = st
            parem_dict["asset"] = focus_roi_diff[st][4]
            algo["parameters"].append(parem_dict)
            
        main_json["algorithms"].append(algo)
                
algo = dict()
algo["algo_name"] = "search_focusbox"
algo["parameters"] = []
for st in final_images:   
    parem_dict = dict()
    parem_dict["state"] = st
    parem_dict["comment"] = st
    parem_dict["focus_box"] = make_focus_box(focus_box[st])
    algo["parameters"].append(parem_dict)
    
main_json["algorithms"].append(algo)        

    
file = open("myjson.json","w+")
file.write(json.dumps(main_json,indent=4))
file.close()

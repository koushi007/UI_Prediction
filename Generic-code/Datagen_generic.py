import cv2
import os
import numpy as np
import pandas as pd
from pandas.core.common import flatten
from utils_app import generate_data,find_icon_max,eval_expr,parse_condition,col_names
import json

## Total data will be stored in this list
data = []

##list of json files of all apps is passed
for json_file in os.listdir('json-dir'):
    ## load the json file
    if not json_file.endswith('.json'):
        continue
    json_file = open('json-dir/'+json_file)
    app_json = json.load(json_file)  
    
    #Read screenshots directory,icon images directory, icon images to states mapping csv
    screenshots_directory = app_json["directory-path"]
    icon_directory = ''
    icom_images_map = ''
    try:
        
        icon_directory = app_json["icon-images-path"]
        icon_images_map = app_json["icon-images-mapping"]
        
    except:
        pass
    ## Types of focus boxes to be detected
    focus_box_types = app_json["focus-boxes"]
    
    ## Mapping of iconstates for the current app
    df_iconstates = pd.read_csv(icon_images_map)
    list_iconstates = list(df_iconstates['Icon_state'].values)
    dict_iconstates = dict(list(df_iconstates.values))
    
    
    ##target file where data to be stored
    app_identifier = app_json["app-identifier"]
    
    ##
    x,y,w,h = -1,-1,-1,-1
    mapping = dict()
    mapping["x"] = x
    mapping["y"] = y
    mapping["w"] = w
    mapping["h"] = h
    ##for each image
    for file in os.listdir(screenshots_directory):
        
        img = cv2.imread(screenshots_directory+file)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        #blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
        ## This will store if any icon present in the required focusbox
        icon_state = ''
        #for each focus box type
        data_row = []
        data_row_icons = [0 for i in range(25)]
        ##searching the icon gloabally with some conditions if required 
        for icon_search in app_json["search-icon-global"]:
            
            ret,thresh = cv2.threshold(gray,icon_search["threshold-value"],255,icon_search["threshold-type"])
            param_ = icon_search["icon-parameters"]
            param = [int(i) for i in param_[:-1]]
            icon_state = find_icon_max(thresh[param[1]:param[1]+param[3],param[0]:param[0]+param[2]],param_[-1],icon_directory)
            
        
        for fbox in focus_box_types:
            
            ##Blurring the image if required
            blur_kernel = fbox["blurring"]
            if blur_kernel > 0:
                img_used = cv2.GaussianBlur(gray,(blur_kernel,blur_kernel),0)
            else:
                img_used = gray
            
            ## Thresholding for better detection of contours
            ret,thresh = cv2.threshold(img_used,fbox["threshold-value"],255,fbox["threshold-type"])
            
            ## Finding all contours for the given thresholding and conditions
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
                ##approximating the contour to a rectangle considering error margin
                approx = cv2.approxPolyDP(cnt, fbox["rectangle-error"]* cv2.arcLength(cnt, True), True)
                if len(approx) == 4 :
                    
                    (x,y,w,h) = cv2.boundingRect(approx)
                    mapping["x"] = x
                    mapping["y"] = y
                    mapping["w"] = w
                    mapping["h"] = h
                    ##parsing the given conditions Ex: "x > 100" will return ["x",">","100"]
                    parsed_conditions = [parse_condition(string) for string in fbox["conditions"]]
                    ##evaluating the given condition expressiong. mapping dict stores mapping of "x" to x
                    evaled_conditions = [eval_expr(mapping[i[0]],i[1],i[2]) for i in parsed_conditions]
                    ##if all the evaluated conditions are true and focux box is at top of hierarchy
                    if all(evaled_conditions) and (hierarchy[0][i][3] == ind or hierarchy[0][i][3]==-1):
                        cv2.rectangle(img, (x,y), (x+w,y+h), tuple(fbox["color"]), 2)
                        outer_contour = [x,y,w,h]
                        data_row += outer_contour
                        if icon_state != '':
                            continue
                        for icon_search in fbox["search-icon"]:
                            parsed_conditions = [parse_condition(string) for string in icon_search["icon-conditions"]]
                            ##evaluating the given condition expressiong. mapping dict stores mapping of "x" to x
                            evaled_conditions = [eval_expr(mapping[i[0]],i[1],i[2]) for i in parsed_conditions]
                            ##if all the evaluated conditions are true and focux box is at top of hierarchy
                            if all(evaled_conditions):
                                param = icon_search["icon-parameters"]
                                parse_param = [parse_condition(string) for string in param[:-1]]
                                eval_param = [eval_expr(mapping[i[0]] if i[0] in ["x","y","w","h"] else i[0],i[1],i[2]) for i in parse_param]
                                icon_state = find_icon_max(img_used[eval_param[1]:eval_param[1]+eval_param[3],eval_param[0]:eval_param[0]+eval_param[2]],param[-1],icon_directory)
                                
        #print(file,"->",icon_state)    
        sz = len(data_row)
        #print(sz)
        for i in range(60-int(sz/4)):
            data_row = data_row + [0,0,720,576]
        try:
            ind = dict_iconstates[icon_state]
            data_row_icons[ind] = 1
        except:
            pass
        data_row = data_row + data_row_icons
        data_row.append(app_identifier)
        data_row.append(file.split('.')[0].split('-')[0])
        data = data+generate_data(data_row,27)
        #data.append(data_row)
    
        cv2.imwrite('new-yt/'+file,img)
        thresh = cv2.cvtColor(thresh,cv2.COLOR_GRAY2BGR)
        #cv2.imwrite('new-nt/thresh'+file,thresh)
 

columns = []
for i in range(60):
    columns = columns + col_names(i+1)
columns = columns + ["Icon"+str(i+1) for i in range(25)]
columns.append('APP')
columns.append('State')
df = pd.DataFrame(2*data,columns = columns)

df.to_csv("Combined_generic_data.csv") 
    




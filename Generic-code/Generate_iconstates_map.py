import cv2
import os
import numpy as np
import pandas as pd
import json

## Total data will be stored in this list
states = []

directory = 'json-traindir/'

##list of json files of all apps is passed
for json_file in os.listdir(directory):
    ## load the json file
    if not json_file.endswith('.json'):
        continue
    json_file = open(directory+json_file)
    app_json = json.load(json_file)  
    
    #Read screenshots directory
    screenshots_directory = app_json["icon-images-path"]
    states_cur_app = []
    try:
        for file in os.listdir(screenshots_directory):
            states_cur_app.append(file.split('.')[0].split('-')[0])
            
        states = list(set(states_cur_app))
        
        df = pd.DataFrame(states,columns = ['iconstate'])
        target_file = app_json['icon-images-mapping']
        df.to_csv(target_file) 
    except:
        pass




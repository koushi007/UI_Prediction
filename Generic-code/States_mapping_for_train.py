import cv2
import os
import numpy as np
import pandas as pd
import json

## Total data will be stored in this list
states = []

##list of json files of all apps is passed
for json_file in os.listdir('json-dir'):
    ## load the json file
    if not json_file.endswith('.json'):
        continue
    json_file = open('json-dir/'+json_file)
    app_json = json.load(json_file)  
    
    #Read screenshots directory
    screenshots_directory = app_json["directory-path"]
    states_cur_app = []
    for file in os.listdir(screenshots_directory):
        states_cur_app.append(file.split('.')[0].split('-')[0])
        
    states = list(set(states_cur_app))
    
    df = pd.DataFrame(states,columns = ['state'])
    target_file = app_json['states-mapping']
    df.to_csv(target_file) 
    




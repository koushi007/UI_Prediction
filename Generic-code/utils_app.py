import operator
import regex as re
import os
import cv2
import numpy as np

ops = {
    '<' : operator.lt,
    '<=' : operator.le,
    '>' : operator.gt,
    '>=' : operator.ge,  
    '==' : operator.eq,
    '!=' : operator.ne,
    '+'  : operator.add,
    '-'  : operator.sub
}


def eval_expr(op1, oper, op2):
    op1, op2 = int(op1), int(op2)
    return ops[oper](op1, op2)

def parse_condition(str):
    x = -1
    try:
        (x,y) = re.search("[><=!+-]=?",str).span()
    except:
        pass
    if x!=-1:
        op1 = str[:x]
        oper = str[x:y]
        op2 = str[y:]
        return [op1.strip(),oper.strip(),op2.strip()]
    else:
        return [str.strip(),'+','0']
def col_names(i):
    return ["x{}".format(i),"y{}".format(i),"w{}".format(i),"h{}".format(i)]

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

def generate_data(data_row,leave):
    data_gen = [data_row]
    for j in range(1,4):
        temp = [ inc(data_row[i],j,720) if i%4==0 else data_row[i] for i in range(len(data_row)-leave)]
        temp = temp + data_row[len(data_row)-leave:]
        data_gen.append(temp)
    
        temp = [inc(data_row[i],j,576) if i%4==1  else data_row[i] for i in range(len(data_row)-leave)]
        temp = temp + data_row[len(data_row)-leave:]
        data_gen.append(temp)
    
        temp = [inc(data_row[i],j,720-data_row[i-2]) if i%4==2  else data_row[i] for i in range(len(data_row)-leave)]
        temp = temp + data_row[len(data_row)-leave:]
        data_gen.append(temp)

        temp = [inc(data_row[i],j,576-data_row[i-2]) if i%4==3   else data_row[i] for i in range(len(data_row)-leave)]
        temp = temp + data_row[len(data_row)-leave:]
        data_gen.append(temp)
        
        temp = [dec(data_row[i],j,0) if i%4==0 else data_row[i] for i in range(len(data_row)-leave)]
        temp = temp + data_row[len(data_row)-leave:]
        data_gen.append(temp)
    
        temp = [dec(data_row[i],j,0) if i%4==1 else data_row[i]  for i in range(len(data_row)-leave)]
        temp = temp + data_row[len(data_row)-leave:]
        data_gen.append(temp)
    
        temp = [dec(data_row[i],j,5) if i%4==2  else data_row[i] for i in range(len(data_row)-leave)]
        temp = temp + data_row[len(data_row)-leave:]
        data_gen.append(temp)

        temp = [dec(data_row[i],j,5) if i%4==3 else data_row[i]  for i in range(len(data_row)-leave)]
        temp = temp + data_row[len(data_row)-leave:]
        data_gen.append(temp)
        
    return data_gen

def compress(prev_data):
    new_data = []
    for data_row in prev_data:
        new_data_row = [ data_row[2*i] | (data_row[2*i+1] << 10) for i in range(int((len(data_row)-1)/2))]
        new_data_row.append(data_row[-1])
        new_data.append(new_data_row)
        
    return new_data

def find_icon(gray_img,start,ref):
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


def find_icon_max(gray_img,start,ref):
    maxi_val = 0
    maxi_image = ''
    for template_file in os.listdir(ref):
        if not template_file.endswith('.png') or not template_file.startswith(start):
            continue
            
        template = cv2.imread(ref+'/'+template_file,0)
        w, h = template.shape[::-1]
        wi, hi = gray_img.shape[::-1]
        
        if w > wi or h > hi:
            continue
        
        res_img = cv2.matchTemplate(gray_img,template,cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res_img)
        
        if max_val > maxi_val:
            maxi_val = max_val
            maxi_image = template_file
        
        
    
    if maxi_val > 0.8:
        #print(maxi_val)
        icon_state = maxi_image[:-4]
        if icon_state[-1] == '2' and icon_state[:4] != "home":
            return icon_state[:-1]
        return icon_state
    
    return ''
    
    
def match_icon(template):
    dist = []
    list_dir = os.listdir(ref)
    for file in list_dir:
        histogram = cv2.calcHist([template], [0],None, [256], [0, 256])
        
        # data1 image
        image = cv2.imread(ref+'/'+file)
        gray_image1 = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        histogram1 = cv2.calcHist([gray_image1], [0],None, [256], [0, 256])
        						
        
        
        c1 = 0
        
        # Euclidean Distace between data1 and test
        i = 0
        while i<len(histogram) and i<len(histogram1):
        	c1+=(histogram[i]-histogram1[i])**2
        	i+= 1
        c1 = c1**(1 / 2)
        dist += list(c1)
        #print(c1)
    print(np.min(dist))
    return list_dir[np.argmin(dist)][:-4] 

# Documentation
## Directories

 

 - [ ] Icon_images
		This folder contains folders of icon images for all apps. 
		
 - [ ] Iconstates-map
		 This folder contains the iconstates mapping csv files for all apps.

 - [ ] json-traindir
	 This folder contains train and preprocessing json files for all apps.
	 
 - [ ] json-testdir
	  This folder contains test json files for all apps.
	  

 - [ ] Outputstates-map
	 This folder contains the output states mapping csv files for all apps.

	  
 - [ ] Training-images
	 This folder contains folders of training images for all apps
	 
 - [ ] Testing-images
	 This folder contains folders of testing images for all apps


## CSV Files

	 

 - [ ] Generic_data.csv
	 This file contains the training data with the following specification.
	 
	 - First column will be row number
	 - Next 4*N columns will have xi,yi,wi,hi values of the ith focus box where N is the number of focus boxes.
	 - Next M columns will be for icon states
	 - Next  column will be app identifier
	 - Next W columns will be output states
	 - The values N,M,W can be found in Global.json file
	 
## Python scripts
 - [ ] Convert_keras_to_tflite.py
		 This file converts the keras model stored in a directory into tflite format.
	

		Line no #4
		converter = tf.lite.TFLiteConverter.from_saved_model(..)
		use the path of model directory.
		
		Line no #8
		with open(.., 'wb') as f:
		use the path to tflite file to be created
	

 - [ ] Generate_iconstates_map
		 Generates the iconstates mapping csv files in the Iconstates-map directory for all apps.
		 
		 Specification:
		 - json-traindir directory should be in the same directory
 - [ ] Generate_iconstates_map
		 Generates the outputstates mapping csv files in the Outputstates-map directory for all apps.
		 
		 Specification:
		 - json-traindir directory should be in the same directory
	 
 - [ ] Datagen_generic.py
	 Generates the training data in Generic_data.csv
	 
		 Specifications: 
		 - Global.json file should be in the same directory	
		 - json-traindir directory should be in the same level and contains the train json files of all apps
 - [ ] Generic_nn_train.py
	 Trains the nueral network model with the train data in Generic_data.csv and stores the model in the directory Generic_{acc} where acc is accuracy of the model.
			
		Specifications:
		- Global.json and Generic_data.csv should be in the same directory
					 

 - [ ] Testing_generic.py
 Test the images in the Testing-images directory.
		 
		 Specifications:
		 - Global.json file should be in the same directory	
		 - json-testdir directory should be in the same level and contains the test json files of all apps
		 - Generic_{acc} model should be in the same directory.



## Json Files

 - [ ] Global.json
			
		"no_iconstates" : 35,  ## number of iconstates
		"no_outputstates" : 45, ##number of outputstates
		"no_focusboxes" : 60, ## number of focusboxes
    
 - [ ] {}-app.json ## Example described here is of Notification app
		These json files are most important part of this project .
			
			"app-name" : "Notification" ## Name of the application
		    "directory-path" : "Training-images/Notification-	Training/", ##Relative path of training images directory
		    "icon-images-path" : "Icon-images/Notification-icon-images/", ##Relative path of icon images directory
		    "icon-images-mapping" : "Iconstates-map/Notification_iconstates.csv", ##Relative path to iconstates mapping csv file
		    "states-mapping" : "Outputstates-map/Notification_statesmap.csv", ## Relative path to outputstates mapping csv file
		    "app-identifier" : 3, ##App identifier is unique to each application

Now to detect focus boxes of different types we need to tune some parameters

	   "blurring" : 0,  
		   ## 0 => using gray image with out blurring
		   ## any odd number(preferably 5 or7) blurs the image with gaussian kernel nxn where n is the given number.
		   ## Note: Even number input should not be given	


       "threshold-value" : 15,
	   "threshold-type" : 1,
			## Adjust these values and run Opencv-generic.py to see how the boxes are being detected
			
       "rectangle-error" : 0.02,
	       ## Look at the links below to understand polygon approximation and error.
	       
       "conditions": [ "w > 100" , "h > 50" , "w<300" ,"y<150"],
       ##Mention conditions to filter the the boxes detected.
       ## Ex: If we know that the box should have a width atleast 50 pixes just mention "w>50". 
       ## Note: The conditons mentioned will be related with an and relation not or relation.
       
       "search-icon" : [
                {
                    "icon-conditions" : [],
                    "icon-parameters" : ["x","y","w","h",""]
                }
            ],

		##If you want to search for an icon in the focus box detected use the search-icon feature
		##Use icon conditions to mention any further conditions like icon is present with "x>100".
		##Icon parameters define the rectangle to search for the icon.
		Ex: With the given icon parameters the algorithm will search for any icon in rectanbel with x,y as its left top point and with width w, height h.
		
       "color": [0,255,0]
	## This is the color of the box generated in the image after running Opencv-generic.py script.
    
	
 

 Know about thresholding here [Opencv Thresholding](https://docs.opencv.org/4.5.2/d7/d4d/tutorial_py_thresholding.html)
 Know about polygon approximation [rectangle-error](https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html)
			
		 

> Written with [StackEdit](https://stackedit.io/).

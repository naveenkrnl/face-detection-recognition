import cv2
import os
import numpy as np	
subject_names = ["", "Naveen","Others"]	
#there is no label 0 in our training data so subject name for index/label 0 is empty

def detect_face_from_image(img):
    
    #convert the test image to gray scale as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    #here we load Face classifier defined in open_Cv libraries, p.s. i use lbpcascade which is faster than others
    face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_default.xml')
    
    #Here i detect all faces from a image
    all_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);
    
    #if there is no face in image it returns None
    if (len(all_faces) == 0):
        return None, None
    
    #we use only first face assuming that image has only one face
    (x, y, w, h) = all_faces[0]
    
    #return the face of image
    return gray[y:y+w, x:x+h], all_faces[0]
	
def prepare_training_data(data_folder_path):
    
    #gets all the directories in training_data folder
    directories = os.listdir(data_folder_path)
    
    #Defined two lists ones for faces and other labels corresponding to each person
    faces = []
    labels = []
    
    #move through every directory and read images in it
    for directory_name in directories:
    
        #our subject directories start with letter 's' so
        #ignore any non-relevant directories if any
        if directory_name.startswith("s"):
            print('I am working')

            #to get the label corresponding to each image we perform: Replace "s1" with "1"
            label = int(directory_name.replace("s", ""))
            
            #to build the path of directory containing images like "training-data/s1"
            subject_directory_path = data_folder_path + "/" + directory_name
            #Get images names using os.listdir
            subject_images_names = os.listdir(subject_directory_path)

            for image_name in subject_images_names:    
                if not image_name.startswith("."):  #to avoid unwanted files          

                    photo_path = subject_directory_path + "/" + image_name
                    
                    #reading image through cv2
                    image = cv2.imread(photo_path)
                    
                    #display an image window to show the image 
                    cv2.imshow("Training on image...", image)
                    cv2.waitKey(100)
                    
                    #detect face
                    face, rect = detect_face_from_image(image)
                    
                    if face is not None:
                        faces.append(face)
                        labels.append(label)
                        
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            cv2.destroyAllWindows()                    
    return faces, labels

print("Preparing data...")
faces, labels = prepare_training_data("training-data")
print("Data prepared")
print("Total faces: ", len(faces))
print("Total labels: ", len(labels))
#create our LBPH face recognizer 
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))
face_cascade = cv2.CascadeClassifier('lbpcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
#initiate id counter
id = 0
# Initialize real-time video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video-width
cam.set(4, 480) # set video-height

minimum_width = 0.1*cam.get(3)
minimum_height = 0.1*cam.get(4)

while True:
    ret, img =cam.read()
    img=cv2.flip(img,1,0) #image in laptops is flipped automatically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)    
    faces_in_image = face_cascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minimum_width), int(minimum_height)),
       )

    for(x,y,w,h) in faces_in_image:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, confidence = face_recognizer.predict(gray[y:y+h,x:x+w])
        print(confidence)
        if (confidence <50):
            id = subject_names[id]
            confidence = " {0}%".format(round(100 - confidence))
        else:
            id = "I Don't Know You"
            confidence = " {0}%".format(round(100 - confidence))

        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)  
    
    cv2.imshow('camera',img) 

    k = cv2.waitKey(10) & 0xff
    if k == 27:         # Press 'ESC' for exiting the program
        break
        
print("I am Cleaning up now")
cam.release()
cv2.destroyAllWindows()
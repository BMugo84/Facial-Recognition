import cv2

cam = cv2.VideoCapture(1)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade =  cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

while cam.isOpened():

    ret, frame = cam.read()

    # convert to grayscale and pass gray to the model
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image=gray, scaleFactor=1.3, minNeighbors=5)
    eyes = eye_cascade.detectMultiScale(image=gray, scaleFactor=1.3, minNeighbors=5)
    

    # draw rectangle 
    # over face
    for fx, fy, fwidth, fheight in faces:
        cv2.rectangle(img=frame, pt1=(fx, fy), pt2=(fx+fwidth, fy+fheight), color=(0,255,0), thickness=2)

    #over eyes
    for ex, ey, ewidth, eheight in eyes:
        cv2.rectangle(img=frame, pt1=(ex, ey), pt2=(ex+ewidth, ey+eheight), color=(255,0,0), thickness=2)


        
    cv2.imshow('Sec Cam', frame)

    if cv2.waitKey(10) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
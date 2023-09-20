import cv2

cam = cv2.VideoCapture(1)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade =  cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

recording = False

# set video height and width 
frame_size = (int(cam.get(3)), int(cam.get(4)))
# init an mp4 codec compressor
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# output the file / write
out = cv2.VideoWriter(filename="video.mp4", fourcc=fourcc, fps=20, frameSize=frame_size)

while cam.isOpened():

    ret, frame = cam.read()

    # convert to grayscale and pass gray to the model
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image=gray, scaleFactor=1.3, minNeighbors=5)
    eyes = eye_cascade.detectMultiScale(image=gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) + len(eyes) > 0:
        recording = True
    
    # write the frame 
    out.write(frame)
        
    cv2.imshow('Sec Cam', frame)

    if cv2.waitKey(10) == ord('q'):
        break

#  quit recording
out.release()
cam.release()
cv2.destroyAllWindows()
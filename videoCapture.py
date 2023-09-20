import cv2
import time
import datetime

cam = cv2.VideoCapture(1)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade =  cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

detection = False
detection_stopped_time = None
timer_started = False
SECONDS_TO_RECORD_AFTER_DETECTION = 5

# set video height and width 
frame_size = (int(cam.get(3)), int(cam.get(4)))
# init an mp4 codec compressor
fourcc = cv2.VideoWriter_fourcc(*'mp4v')


while cam.isOpened():

    ret, frame = cam.read()

    # convert to grayscale and pass gray to the model
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(image=gray, scaleFactor=1.3, minNeighbors=5)
    eyes = eye_cascade.detectMultiScale(image=gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) + len(eyes) > 0:
        if detection:
            timer_started = False
        else:
            detection = True
            # output the file / write
            current_time = datetime.datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            out = cv2.VideoWriter(filename=f"{current_time}.mp4", fourcc=fourcc, fps=20, frameSize=frame_size)
            print("Started Recording!")
    elif detection:
        if timer_started:
            if time.time() - detection_stopped_time >= SECONDS_TO_RECORD_AFTER_DETECTION:
                detection = False
                timer_started = False
                out.release()
                print("Stop Recording!")
        else:
            timer_started = True
            detection_stopped_time = time.time()

    # write the frame 
    if detection:
        out.write(frame)
        
    cv2.imshow('Sec Cam', frame)

    if cv2.waitKey(10) == ord('q'):
        break

#  quit recording
out.release()
cam.release()
cv2.destroyAllWindows()
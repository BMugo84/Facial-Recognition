import cv2

cap  = cv2.VideoCapture(1)
cv2.namedWindow('Image Collection', cv2.WINDOW_NORMAL)

while cap.isOpened():
    ret, frame = cap.read()

    cv2.imshow('Image Collection', frame)

    if cv2.waitKey(10) & 0XFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
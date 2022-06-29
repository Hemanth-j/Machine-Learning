import cv2
face_cascade = cv2.CascadeClassifier('C:\\Users\\HEMANTH\\Downloads\\haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('C:\\Users\\HEMANTH\\Downloads\\haarcascade_eye.xml')
capture = cv2.VideoCapture(0)
while (True):
    ret, frame = capture.read()
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayFrame,scaleFactor= 1.05,minNeighbors= 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255,0, 0), 3)
    eyes = eye_cascade.detectMultiScale(grayFrame, scaleFactor=1.05, minNeighbors=5)
    for (x, y, w, h) in eyes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (56, 0, 255), 3)
    cv2.imshow('video gray', frame)
    if cv2.waitKey(1) == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()

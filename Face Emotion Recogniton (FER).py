from fer import FER
import cv2
detector=FER(mtcnn=True)
cap = cv2.VideoCapture(0)
while (True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    emotio, score = detector.top_emotion(frame)
    print(emotio, score)
cap.release()
cv2.destroyAllWindows()
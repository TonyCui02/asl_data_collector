import cv2 as cv2
import numpy as np

cap = cv2.VideoCapture(0)
save_frames = 1
num_frames = 0
element = 'F'

while(True):
    ret, frame = cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # frame = cv2.resize(frame, (224, 224))
    
    low = np.array([0, 20, 70])
    high = np.array([20, 255, 255])

    frame = cv2.inRange(frame, low, high)

    kernel = np.ones((2,2), np.uint8)
    frame = cv2.dilate(frame, kernel, iterations=4)
    frame = cv2.GaussianBlur(frame, (7,7), 100)


    copy = frame.copy()
    cv2.putText(copy, "current frame: {}".format(num_frames), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2 )
    cv2.imshow("frame", copy)

    if save_frames == 1 and num_frames > 60 and num_frames <= 160:
        cv2.imwrite('C:/Users/Tonyc/Repos/Tfjs_models/Asl_data_collector/Train/{}/{}.jpg'.format(element, num_frames), frame)

    elif num_frames > 160 and num_frames <= 200:
        cv2.imwrite('C:/Users/Tonyc/Repos/Tfjs_models/Asl_data_collector/Test/{}/{}.jpg'.format(element, num_frames), frame)



    num_frames += 1

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
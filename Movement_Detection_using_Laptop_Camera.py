
########## __________ Movement detection using laptop camera
########## __________ 
########## __________ Importing the necessary libraries
import cv2
import numpy as np

########## __________ Capturing frames from laptop camera
cap = cv2.VideoCapture(1)

########## __________ Getting the width and the height of the captured frame
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

########## __________ Preparing the output file and save it into the same 
########## __________ directory with the name "output.avi"

out = cv2.VideoWriter("output.avi", fourcc, 5.0, (1280, 720))

########## __________ Reading double frames for the purpose of comparing
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    ##### _____ Finding the difference between the two frames using absolute subtraction
    diff = cv2.absdiff(frame1, frame2)
    ##### _____ process of changing the difference into a visual frame
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)
    ##### _____ Detect the contour of the moving areas
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    ##### _____ detecting an dprocess each contour sperately
    for contour in contours:
        ##### _____ specifying the coordinates of each contour
        (x, y, w, h) = cv2.boundingRect(contour)

        ##### _____ thresholding the contours according to their areas
        if cv2.contourArea(contour) < 1500:
            continue
        
        ##### _____ drawing rectangles arround the detected movements
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        ##### _____ outputing a status notifies that there is a movement
        cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 3)

    ##### _____ resizing the output frame according the size of the used screen
    image = cv2.resize(frame1, (1280, 720))
    out.write(image)
    ##### _____ Visualling the frame to see the result in real-time
    cv2.imshow("feed", frame1)
    frame1 = frame2
    ##### _____ Read the next frame
    ret, frame2 = cap.read()
    if cv2.waitKey(40) == 27:
        break

cv2.destroyAllWindows()
cap.release()
out.release()

import cv2
from simple_facerec import SimpleFacerec
import numpy as np

sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

cap = cv2.VideoCapture(0)
vid = cv2.VideoCapture("videos/v2.mp4")
vid2 = cv2.VideoCapture("videos/carry.mp4")
vidBeast = cv2.VideoCapture("videos/vidBeast.mp4")
vidSir = cv2.VideoCapture("videos/vidSir.mp4")
# some = cv2.VideoCapture("")

while True:
    ret, frame = cap.read()

    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        # print(int(x2)-int(x1), int(y2)-int(y1))
        # print(int(x2), int(x1), int(y2), int(y1))
        # print(type(x1.item()))
        # points = np.array([[x1,y1],[x2,y2],[x2,y1],[x1,y2]])
        # cv2.fillPoly(frame, pts=[points], color=(255,255,255))

        # cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        if name not in ["Unknown", "", " "]:
            # masknew = np.zeros((frame[0],frame[1]), np.uint8)
            # maskInverted = cv2.bitwise_not(frame)
            # imgaug = cv2.bitwise_not(imgaug, imgaug, mask=maskInverted)
            # cv2.imshow('masknew', maskInverted)

            if name == "BB Ki Vines":
                video = vid
            elif name == "CarryMinati":
                video = vid2
            elif name == "MrBeast":
                video = vidBeast
            elif name == "sir":
                video = vidSir
            # elif name == "linus torvalds":
            #     video = some
            else:
                video = vid

            ret, frame2 = video.read()

            xwidth=abs(x2.item()-x1.item());
            ywidth=abs(y2.item()-y1.item());

            frame2=cv2.resize(frame2,(xwidth,ywidth),interpolation = cv2.INTER_CUBIC)

            winname = "frame2"
            cv2.namedWindow(winname)

            cv2.moveWindow(winname, x1, y1)
            print(type(frame2), np.shape(frame2))
            cv2.imshow(winname, frame2)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), -1)
            cv2.putText(frame, name,(x1, y1 - 20), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
        else:
            cv2.destroyWindow('Frame2')
            cv2.putText(frame, name,(x1, y1 - 20), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()

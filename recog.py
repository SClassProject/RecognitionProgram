# face_recog.py

import os, time, math
import numpy as np
import cv2
import uuid
import face_recognition
import mediapipe as mp
from datetime import datetime
import camera
from id import id, room_id
from client_socket import sio, connect, disconnect, attend, raiseHand

# 손 관련 데이터
compareIndex = [[18,4], [6,8], [10,12], [14,16], [18,20]]
open = [False, False, False, False, False]
gesture = [[True, True, True, True, True, "Hand!"]]
str = ''

start = datetime.now()
# start = start.minute
start = start.second

sio.connect('https://192.168.71.132:443', namespaces='/room')

class Recog():
    def __init__(self):
        self.camera = camera.VideoCamera()

        # 손
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands

        # 얼굴
        self.known_face_encodings = []
        self.known_face_names = []

        dirname = 'img'
        files = os.listdir(dirname)
        for filename in files:
            name, ext = os.path.splitext(filename)
            if ext == '.jpg':
                self.known_face_names.append(name)
                pathname = os.path.join(dirname, filename)
                img = face_recognition.load_image_file(pathname)
                face_encoding = face_recognition.face_encodings(img)[0]
                self.known_face_encodings.append(face_encoding)

        self.face_locations = []
        self.face_encodings = []
        self.face_names = []
        self.process_this_frame = True

    def __del__(self):
        del self.camera

    def get_frame(self):
        frame = self.camera.get_frame()

        global start
        current = datetime.now()
        # current = current.minute
        current = current.second

        with self.mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
            h, w, c = frame.shape

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.flip(frame, 1)
            frame.flags.writeable = False
            results = hands.process(frame)
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 손
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    for i in range(0, 5):
                        open[i] = self.dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[compareIndex[i][0]].x, handLms.landmark[compareIndex[i][0]].y) < self.dist(handLms.landmark[0].x, handLms.landmark[0].y, handLms.landmark[compareIndex[i][1]].x, handLms.landmark[compareIndex[i][1]].y)
                    
                    # print(open)   #손 동작 확인

                    text_x = (handLms.landmark[0].x * w)
                    text_y = (handLms.landmark[0].y * h)

                    for i in range(0, len(gesture)) :
                        flag = True
                        for j in range(0, 5) :
                            if (gesture[i][j] != open[j]) :
                                flag = False
                        if (flag == True) :
                            cv2.putText(frame, gesture[i][5], (round(text_x)- 50, round(text_y) -250), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 0), 4)
                            str = id + ":hand"
                            print(str)
                            raiseHand(str)
                            time.sleep(3)

                    self.mp_drawing.draw_landmarks(frame, handLms, self.mp_hands.HAND_CONNECTIONS)


                for num, hand in enumerate(results.multi_hand_landmarks):
                    self.mp_drawing.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS, 
                                            self.mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                            self.mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                            )

            # 얼굴
            if self.process_this_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                rgb_small_frame = small_frame[:, :, ::-1]
                
                self.face_locations = face_recognition.face_locations(rgb_small_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    min_value = min(distances)

                    name = "Unknown"
                    if min_value < 0.6:
                        index = np.argmin(distances)
                        name = self.known_face_names[index]

                    self.face_names.append(name)

                    if name == id and abs(current - start) > 5:
                        str = id + ":attend"
                        attend(str)
                        print(str)
                        start = datetime.now()
                        # start = start.minute
                        start = start.second


            self.process_this_frame = not self.process_this_frame

            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        return frame

    def get_jpg_bytes(self):
        frame = self.get_frame()
        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()

    def dist(self, x1, y1, x2, y2) :
        return math.sqrt(math.pow(x1-x2, 2)) + math.sqrt(math.pow(y1-y2, 2))



if __name__ == '__main__':
    recog = Recog()
    print(recog.known_face_names)
    while True:
        frame = recog.get_frame()

        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            sio.emit('leave', {'id': id, 'room_id' : room_id}, namespace='/room')
            sio.disconnect()
            break

    cv2.destroyAllWindows()
    print('finish')

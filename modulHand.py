import cv2 as cv
import mediapipe as mp
import time 
import math
class handDetect():
    def __init__(self,mode = True,maxhands = 4,minDetectConf = 0.5,maxTrackingConf = 0.5):
        self.mode = mode
        self.maxhands = maxhands
        self.minDetectConf = minDetectConf
        self.maxTrackingConf = maxTrackingConf
        self.mpHands = mp.solutions.hands 
        self.hands = self.mpHands.Hands(
            static_image_mode = self.mode,
            max_num_hands = self.maxhands,
            min_detection_confidence = self.minDetectConf,
            min_tracking_confidence = self.maxTrackingConf
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.index_up = False
        self.middel_up = False
        self.ring_up = False
        self.pinky_up  = False
        self.first_up = False
        self.index_only_gesture = False
        


    def detector(self,img,Draw = False):
        imageRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        if self.results.multi_hand_landmarks:
            for handlm in self.results.multi_hand_landmarks:
                if Draw:
                    self.mpDraw.draw_landmarks(img,handlm,self.mpHands.HAND_CONNECTIONS,
                                           self.mpDraw.DrawingSpec((0,0,255),thickness = 2,circle_radius = 2),
                                           self.mpDraw.DrawingSpec((0,255,0),thickness = 2,)) 
        return img
    def handposition(self,img):
        allhands = []
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                hannddd = []
                for id,lm in enumerate(hand.landmark):
                    h,w,c = img.shape
                    cx,cy= int(lm.x * w),int(lm.y *h)
                    hannddd.append([id,cx,cy])
                allhands.append(hannddd)
        return allhands
    def how(self):
        if self.results.multi_hand_landmarks:
            return len(self.results.multi_hand_landmarks)
        return 0
    def exitScreen(self,img):
        if self.results.multi_hand_landmarks:
            for handlm in self.results.multi_hand_landmarks:
                index_pip = None
                index_tip = None
                first_pip = None
                first_tip = None
                middle_pip = None
                middle_tip = None
                ring_pip= None
                ring_tip = None
                pinky_pip = None
                pinky_tip = None
                for id,lm in enumerate(handlm.landmark):
                    h,w,c = img.shape
                    if id == 6:
                        cy = int(lm.y * h)
                        index_pip = cy
                    if id == 8:
                        cy = int(lm.y * h)
                        index_tip = cy
                    if id == 2:
                        cy = int(lm.y * h)
                        first_pip = cy
                    if id == 4:
                        cy = int(lm.y * h)
                        first_tip = cy
                    if id == 10:
                        cy = int(lm.y * h)
                        middle_pip = cy
                    if id == 12:
                        cy = int(lm.y * h)
                        middle_tip = cy 
                    if id == 14:
                        cy = int(lm.y * h)
                        ring_pip = cy
                    if id == 16:
                        cy = int(lm.y * h)
                        ring_tip = cy
                    if id == 18:
                        cy = int(lm.y * h)
                        pinky_pip = cy
                    if id == 20:
                        cy = int(lm.y * h)
                        pinky_tip = cy
                
                self.first_up = first_pip > first_tip
                
                self.index_up = index_pip > index_tip

                self.middel_up = middle_pip >middle_tip
                
                self.ring_up = ring_pip>ring_tip
                
                self.pinky_up = pinky_pip>pinky_tip

            if self.pinky_up and not self.ring_up and not self.middel_up  and self.index_up  and self.first_up:
                print("🤟 detected")
                return True 
            return False
    def reset(self,img):
        if self.results.multi_hand_landmarks:
            for handlm in self.results.multi_hand_landmarks:
                index_pip = None
                index_tip = None
                first_pip = None
                first_tip = None
                middle_pip = None
                middle_tip = None
                ring_pip= None
                ring_tip = None
                pinky_pip = None
                pinky_tip = None
                for id,lm in enumerate(handlm.landmark):
                    h,w,c = img.shape
                    if id == 6:
                        cy = int(lm.y * h)
                        index_pip = cy
                    if id == 8:
                        cy = int(lm.y * h)
                        index_tip = cy
                    if id == 2:
                        cy = int(lm.y * h)
                        first_pip = cy
                    if id == 4:
                        cy = int(lm.y * h)
                        first_tip = cy
                    if id == 10:
                        cy = int(lm.y * h)
                        middle_pip = cy
                    if id == 12:
                        cy = int(lm.y * h)
                        middle_tip = cy 
                    if id == 14:
                        cy = int(lm.y * h)
                        ring_pip = cy
                    if id == 16:
                        cy = int(lm.y * h)
                        ring_tip = cy
                    if id == 18:
                        cy = int(lm.y * h)
                        pinky_pip = cy
                    if id == 20:
                        cy = int(lm.y * h)
                        pinky_tip = cy
                def finguerUp(pip,tip):
                    return abs(pip>tip) > 15
                
                self.first_up = finguerUp(first_pip,first_tip) 
                
                self.index_up = index_pip > index_tip

                self.middel_up = middle_pip >middle_tip
                
                self.ring_up = ring_pip>ring_tip
                
                self.pinky_up = pinky_pip>pinky_tip

            if not self.pinky_up and not self.ring_up and  self.middel_up  and self.index_up  and not self.first_up:
                print("✌️ detected")
                return True 
            return False
        
    def switch(self,img):
        if self.results.multi_hand_landmarks:
            for handlm in self.results.multi_hand_landmarks:
                index_pip = None
                index_tip = None
                first_pip = None
                first_tip = None
                middle_pip = None
                middle_tip = None
                ring_pip= None
                ring_tip = None
                pinky_pip = None
                pinky_tip = None
                for id,lm in enumerate(handlm.landmark):
                    h,w,c = img.shape
                    if id == 6:
                        cy = int(lm.y * h)
                        index_pip = cy
                    if id == 8:
                        cy = int(lm.y * h)
                        index_tip = cy
                    if id == 2:
                        cy = int(lm.y * h)
                        first_pip = cy
                    if id == 4:
                        cy = int(lm.y * h)
                        first_tip = cy
                    if id == 10:
                        cy = int(lm.y * h)
                        middle_pip = cy
                    if id == 12:
                        cy = int(lm.y * h)
                        middle_tip = cy 
                    if id == 14:
                        cy = int(lm.y * h)
                        ring_pip = cy
                    if id == 16:
                        cy = int(lm.y * h)
                        ring_tip = cy
                    if id == 18:
                        cy = int(lm.y * h)
                        pinky_pip = cy
                    if id == 20:
                        cy = int(lm.y * h)
                        pinky_tip = cy
                def finguerUp(pip,tip):
                    return abs(pip>tip) > 15
                
                self.first_up = finguerUp(first_pip,first_tip) 
                
                self.index_up = index_pip > index_tip

                self.middel_up = middle_pip >middle_tip
                
                self.ring_up = ring_pip>ring_tip
                
                self.pinky_up = pinky_pip>pinky_tip

            if  self.pinky_up and  self.ring_up and  self.middel_up  and  not self.index_up  and not self.first_up:
                print("👌 detected")
                return True 
            return False
        

    
                

                    
def main():
    ctime = 0
    ptime = 0
    detect = handDetect()
    cap = cv.VideoCapture(0)
    while True:
        succes , img = cap.read()
        img = detect.detector(img)

        hands  = detect.handposition(img)
        #if  hands:
            #print(f"hand  {hands[0][20]}")
        
        if detect.exitScreen(img):
            cap.release()
            cv.destroyAllWindows()
            break

        
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        cv.putText(img,str(int(fps)),(12,60),cv.FONT_HERSHEY_PLAIN,3,(0,255,0),thickness=2)
        cv.imshow("image",img)
        cv.waitKey(1)
if __name__ == "__main__":
    main()
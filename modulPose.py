import cv2 as cv 
import time as time 
import mediapipe as mp
import math
import modulHand
class PoseDetect():
    def __init__(self,mode = False,modelComlex = 1,smootLandmarks = True,enabaleSegmentation = False,smoothSegmentation = True ,minDetectionConf = 0.5,minTrackingConf = 0.5):
        self.mode = mode
        self.modelComlex = modelComlex
        self.smootLandmarks = smootLandmarks
        self.enabaleSegmentation = enabaleSegmentation
        self.smoothSegmentation = smoothSegmentation
        self.minDetectionConf = minDetectionConf
        self.minTrackingConf = minTrackingConf
        self.mpPose = mp.solutions.pose

        self.pose = self.mpPose.Pose(
            static_image_mode=self.mode,
            model_complexity=self.modelComlex,
            smooth_landmarks=self.smootLandmarks,
            enable_segmentation=self.enabaleSegmentation,
            smooth_segmentation=self.smoothSegmentation,
            min_detection_confidence=self.minDetectionConf,
            min_tracking_confidence=self.minTrackingConf
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.count = 0
        self.stage = None

        self.timeReset = 2
        self.lastTimeReset = 0
        self.lastTimeSwich = 0
        self.showMessageReset = False
        self.timeShowMessage = 1.5

        self.showMessageSwitch = False
        self.exercice = 0
        self.exerciceName = "Biceps"
        


    def trackPose(self,img):
        imageRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        self.results = self.pose.process(imageRGB)
        if self.results.pose_landmarks:
            self.mpDraw.draw_landmarks(img,self.results.pose_landmarks,self.mpPose.POSE_CONNECTIONS,
                                       self.mpDraw.DrawingSpec(color = (0,0,255),thickness = -1,circle_radius = 3),
                                       self.mpDraw.DrawingSpec(color = (0,255,0),thickness = 2))
    def trackNose(self,img):
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                cx,cy = int(w *lm.x),int(h*lm.y)
                if id == 0:
                    cv.circle(img,(cx,cy),3,(255,0,0),thickness=-1)
    def calulateBicep(self,img):
        shoulder = []
        elbow = []
        wrist = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c = img.shape
                if id == 12:
                    cx,cy = int(w*lm.x),int(h*lm.y)
                    shoulder = [cx,cy]
                if id == 14:
                    cx,cy = int(w*lm.x),int(h*lm.y)
                    elbow = [cx,cy]
                if id == 16:
                    cx,cy = int(w*lm.x),int(h*lm.y)
                    wrist = [cx,cy]
            v1 = [elbow[0]-shoulder[0],elbow[1]-shoulder[1]]
            v2 = [elbow[0]-wrist[0],elbow[1]-wrist[1]]
            dot = v1[0] * v2[0] + v1[1] * v2[1] 
            v1_D = math.sqrt(v1[0] **2 + v1[1]**2)
            v2_D = math.sqrt(v2[0] **2 + v2[1]**2)
            angle = math.degrees(math.acos(dot / (v1_D * v2_D)))
            
            if angle <120:
                self.stage = "Up"
            elif angle >160 and self.stage == "Up":
                self.stage = "Down"
                self.count +=1


            cv.putText(img, self.stage, (360, 70), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
            cv.putText(img, f"Reps: {self.count}", (360, 120), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
            cv.putText(img, self.exerciceName, (360, 190), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
            
    def calculateSquat(self, img):
        right_hip = []
        right_knee = []
        right_ankle = []
        if self.results.pose_landmarks:
            for id ,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c  = img.shape
                if id == 24:
                    cx, cy = int(w * lm.x),int(h*lm.y)
                    right_hip = [cx,cy]
                if id == 26:
                    cx, cy = int(w * lm.x),int(h*lm.y)
                    right_knee = [cx,cy]
                if id == 28:
                    cx, cy = int(w * lm.x),int(h*lm.y)
                    right_ankle = [cx,cy]
            v1 = [right_knee[0] - right_hip[0], right_knee[1] - right_hip[1]]
            v2 = [right_knee[0] - right_ankle[0], right_knee[1] - right_ankle[1]]

            dot = v1[0] * v2[0] + v1[1] * v2[1]
            v1_sqrt = math.sqrt(v1[0] ** 2 + v1[1]**2)
            v2_sqrt = math.sqrt(v2[0] ** 2 + v2[1]**2)

            angle = math.degrees(math.acos(dot / (v1_sqrt * v2_sqrt)))
            if angle <= 120:
                self.stage = "Down"
            elif angle > 160 and self.stage == "Down":
                self.stage = "Up"
                self.count +=1
            

            cv.putText(img, self.stage, (360, 70), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
            cv.putText(img, f"Reps: {self.count}", (360, 120), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
            cv.putText(img, self.exerciceName, (360, 190), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)


    def squat(self,img):
        
        if self.results.pose_landmarks:
            hip = None
            knee = None

            

            for id ,lm in enumerate(self.results.pose_landmarks.landmark):
                h,w,c  = img.shape
                if id == 24:
                    cx, cy = int(w * lm.x),int(h*lm.y)
                    hip = [cx,cy]
                elif id == 26:
                    cx, cy = int(w * lm.x),int(h*lm.y)
                    knee = [cx,cy]
            
            
            if hip[1]> knee[1]:
                self.stage = "Down"
            elif hip[1]<= knee[1] and self.stage == "Down":
                self.stage = "Up" 
                self.count +=1 
            
                
            cv.putText(img, f"Reps: {self.count}", (360, 120), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
            cv.putText(img, self.stage, (360, 70), cv.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)
                
         
def main():
    ctime = 0
    
    ptime = 0
    posss = PoseDetect()
    cap = cv.VideoCapture(0)
    hand = modulHand.handDetect()
    while True:
        succes,img = cap.read()
        img = hand.detector(img)
        posss.trackPose(img)
        if hand.exitScreen(img):
            cap.release()
            cv.destroyAllWindows()
            break

        if hand.reset(img):
            currentTime  = time.time()
            if currentTime - posss.lastTimeReset >posss.timeReset:
                posss.count = 0
                posss.showMessageReset = True 
                posss.lastTimeReset = currentTime
        if posss.showMessageReset:
            if time.time()-posss.lastTimeReset < posss.timeShowMessage:
                cv.putText(img, "RESET!", (200,200), cv.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
            else:
                posss.showMessageReset = False
        
        
        
        if posss.exercice == 0:
            posss.calulateBicep(img)
        elif posss.exercice == 1:
            posss.calculateSquat(img)
        
        if hand.switch(img):
            currentTime = time.time()
            if currentTime-posss.lastTimeSwich >posss.timeReset:
                if posss.exercice == 0:
                    posss.exercice = 1
                    posss.exerciceName = "Squats"
                    posss.lastTimeSwich = currentTime
                    posss.showMessageSwitch = True
                    
                elif posss.exercice == 1:
                    posss.exercice = 0
                    posss.exerciceName =  "Biceps"
                    posss.showMessageSwitch = True
                    posss.lastTimeSwich = currentTime
                    
        
        if posss.showMessageSwitch :
            if time.time()-posss.lastTimeSwich <posss.timeShowMessage:
                cv.putText(img, f"Switch to {posss.exerciceName} ", (70,200), cv.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)
            else:
                posss.showMessageSwitch= False            
                

        




        
        



        #posss.calulateBicep(img)
        
        #posss.calulateBicep(img)
        #posss.calculateSquat(img)
        #posss.squat(img)

        
        ctime = time.time()
        fps = 1/(ctime-ptime)
        ptime = ctime
        cv.putText(img,str(int(fps)),(20,60),cv.FONT_HERSHEY_DUPLEX,3,(0,255,0),thickness=2)
        cv.imshow("img",img)
        hand

        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    cv.destroyAllWindows()



if __name__ == "__main__":
    main()
           
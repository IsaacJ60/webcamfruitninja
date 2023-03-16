import cv2
import mediapipe as mp
import pygame as py

WIDTH, HEIGHT = 800, 600
screen = py.display.set_mode((WIDTH, HEIGHT))

# x and y index
XI, YI = 1, 2

RED = (255, 0, 0)
GREY = (127, 127, 127)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)


class HandTracker:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, modelComplexity=1, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    
    def handsFinder(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image
    
    def positionFinder(self, image, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            Hand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(Hand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmlist.append([id, cx, cy])
        
        return lmlist


def main():
    # get video from webcam
    cap = cv2.VideoCapture(0)
    tracker = HandTracker()
    
    while True:
        success, image = cap.read()
        image = tracker.handsFinder(image)
        pointLocations = tracker.positionFinder(image)
        
        image = cv2.flip(image, 1)
        
        cv2.imshow("Video", image)
        cv2.waitKey(1)
        
        if len(pointLocations) > 0:
            # get pos and stuff
            detectMovement(pointLocations)


def detectMovement(pointLocations):
    detectWrist(pointLocations)


def detectWrist(pointLocations):
    print(pointLocations[0][XI], pointLocations[0][YI])


running = True

while running:
    for evt in py.event.get():
        if evt.type == py.QUIT:
            running = False
    
    mx, my = py.mouse.get_pos()
    mb = py.mouse.get_pressed()
    
    main()
    
    py.display.flip()

quit()

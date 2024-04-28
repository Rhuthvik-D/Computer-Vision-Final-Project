import cv2
import imutils
import numpy as np

import tkinter as tk 
from tkinter import * 
from tkinter import messagebox as mb 

# Snake game in Python

score = 0
max_score = 3
list_capacity = 0
max_lc = 20
l = []
flag = 0
apple_x = None
apple_y = None
center = None
hurdles = []


#Defining functions to read hurdle image
def read_image(path):
    original_image = cv2.imread(path)
    resized_image = cv2.resize(original_image, (20, 20))
    return resized_image


# Define hurdle positions
# Function to generate random apple and hurdle positions
def generate_hurdles(frame):
    num_hurdles = 4
    global hurdles
    if not hurdles:
        # Generate random hurdle positions
        hurdles = [(np.random.randint(30, frame.shape[1] - 30), np.random.randint(30, frame.shape[0] - 30)) for _ in range(num_hurdles)]

# distance function
def dist(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

# Load hurdle image
hurdle_image = read_image('hurdle_image.png')  # Provide the path to your hurdle image


cap = cv2.VideoCapture(0)
# cap.set(3, 1280)
# cap.set(4, 720)


res = 'no'

while 1:
    ret, frame = cap.read()
    img = imutils.resize(frame.copy(), width=600)
    img = cv2.GaussianBlur(img, (11, 11), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Generate random apple and hurdle positions if needed
    generate_hurdles(frame)

    if apple_x is None or apple_y is None:
        # assigning random coefficients for apple coordinates
        apple_x = np.random.randint(30, frame.shape[0] - 30)
        apple_y = np.random.randint(100, 350)

    cv2.circle(frame, (apple_x, apple_y), 3, (0, 0, 255), -1)

    # Display hurdles
    for hurdle_pos in hurdles:
        # Draw hurdle image at hurdle position
        hurdle_x, hurdle_y = hurdle_pos
        frame[hurdle_y:hurdle_y + hurdle_image.shape[0], hurdle_x:hurdle_x + hurdle_image.shape[1]] = hurdle_image

    # change this range according to your need
    greenLower = (29, 86, 18)
    greenUpper = (93, 255, 255)

    # masking out the green color
    mask = cv2.inRange(img, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    if len(cnts) > 0:
        ball_cont = max(cnts, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(ball_cont)

        M = cv2.moments(ball_cont)
        center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))

        if radius > 10:
            cv2.circle(frame, center, 2, (0, 0, 255), 3)

            if len(l) > list_capacity:
                l = l[1:]

            if prev_c and (dist(prev_c, center) > 3.5):
                l.append(center)

            apple = (apple_x, apple_y)
            if dist(apple, center) < 5:
                score += 1
                if score == max_score:
                    flag = 1
                list_capacity += 1
                apple_x = None
                apple_y = None

    for i in range(1, len(l)):
        if l[i - 1] is None or l[i] is None:
            continue
        r, g, b = np.random.randint(0, 255, 3)

        cv2.line(frame, l[i], l[i - 1], (int(r), int(g), int(b)), thickness=int(len(l) / max_lc + 2) + 2)

    cv2.putText(frame, 'Score :' + str(score), (450, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 203), 2)
    if flag == 1:
        cv2.putText(frame, 'YOU WIN !!', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 3)
        cv2.imshow('live feed', frame)
        res = mb.askquestion('Exit Application', 'Retry?')       
        if res == 'yes' : 
            score = 0
            list_capacity = 0
            max_lc = 20
            l = []
            flag = 0
            center = None
            res = 'no'
            continue
              
        else : 
            cv2.waitKey(1000) 
            break
            #mb.showinfo('Return', 'Returning to main application') 
            # Delay before closing the window
        

    # Check collision with hurdles
    for hurdle_pos in hurdles:
        if center is not None and dist(center, hurdle_pos) < 15:
            flag = -1
            break

    # Game over condition
    if flag == -1:
        cv2.putText(frame, 'GAME OVER !!', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
        cv2.imshow('live feed', frame)
        res = mb.askquestion('Exit Application', 'Retry?')       
        if res == 'yes' : 
            score = 0
            list_capacity = 0
            max_lc = 20
            l = []
            flag = 0
            center = None
            res = 'no'
            continue   
        else : 
            cv2.waitKey(1000) 
            break

    cv2.imshow('live feed', frame)
    cv2.imshow('mask', mask)

    prev_c = center

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
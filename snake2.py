import cv2
import imutils
import numpy as np
import math
from tkinter import messagebox as mb
from random import randint,shuffle

# Snake game in Python

WIN_SCORE = 10
MAX_OBSTACLES = 5
MIN_DISTANCE_BW_OBSTACLES = 60
MIN_DISTANCE_FOOD_OBSTACLE = 60
LENGTH_INCREASE_FROM_FOOD = 20
STARTING_LENGTH = 70
SNAKE_COLOR = (0, 0, 255)
SNAKE_THICKNESS = 12
GREEN_LOWER_THRESHOLD = (29, 86, 18)
GREEN_UPPER_THRESHOLD = (93, 255, 255)
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

# Distance function
def distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

# Check if a point on overlaps an image on the frame
def checkPointOverlapImage(pointLocation, imageLocation, imageSize):
    return (imageLocation[0] - imageSize[0]//2 < pointLocation[0] < imageLocation[0] + imageSize[0]//2 and \
        imageLocation[1] - imageSize[1]//2 < pointLocation[1] < imageLocation[1] + imageSize[1]//2)

# Draw image over frame
def drawImageOverFrame(frame, image, imageLocation):
    imageSize = image.shape[1], image.shape[0]
    frame[(imageLocation[1] - (imageSize[1]//2)) : (imageLocation[1] + (imageSize[1]//2)), 
            (imageLocation[0] - (imageSize[0]//2)) : (imageLocation[0] + (imageSize[0]//2))] = image

class SnakeGame:
    def __init__(self, foodImagePath, obstacleImagePath) -> None:
        self.points = []
        self.lengths = []
        self.totalLength = STARTING_LENGTH
        self.currentLength = 0
        self.obstacleLocations = []
        self.obstacleCount = MAX_OBSTACLES
        self.previousHead = 0, 0
        self.score = 0
        self.gameOver = False

        self.obstacleImage = cv2.imread(obstacleImagePath, cv2.IMREAD_UNCHANGED)
        self.obstacleImageSize = self.obstacleImage.shape[1], self.obstacleImage.shape[0]
        self.generateObstacles()

        self.foodImage = cv2.imread(foodImagePath, cv2.IMREAD_UNCHANGED)
        self.foodImageSize = self.foodImage.shape[1], self.foodImage.shape[0]
        self.foodLocation = 0, 0
        self.generateRandomFoodLocation()
    
    def generateObstacles(self):
        self.obstacleLocations = []
        while len(self.obstacleLocations) < self.obstacleCount:
            temp_point = (randint(50, VIDEO_WIDTH-50), randint(50, VIDEO_HEIGHT-50))
            if all(distance(temp_point,p) >= MIN_DISTANCE_BW_OBSTACLES for p in self.obstacleLocations):
                self.obstacleLocations.append(temp_point)

    def generateRandomFoodLocation(self):
        while True:
            self.foodLocation = randint(50, VIDEO_WIDTH-50), randint(50, VIDEO_HEIGHT-50)
            if all(distance(self.foodLocation, obstacle) > MIN_DISTANCE_FOOD_OBSTACLE for obstacle in self.obstacleLocations):
                break
    
    def drawObjects(self, image):
        # Draw snake
        if self.points:
            for i, _ in enumerate(self.points):
                if i != 0:
                    cv2.line(image, self.points[i], self.points[i-1], SNAKE_COLOR, SNAKE_THICKNESS)
            cv2.circle(image, self.points[-1], int(SNAKE_THICKNESS/2), SNAKE_COLOR, cv2.FILLED)
        
        # Draw food
        drawImageOverFrame(image, self.foodImage, self.foodLocation)

        # Draw obstacles
        for location in self.obstacleLocations:
            drawImageOverFrame(image, self.obstacleImage, location)
        
        return image

    def reset(self):
        self.points = []
        self.lengths = []
        self.totalLength = STARTING_LENGTH
        self.currentLength = 0
        self.obstacleLocations = []
        self.obstacleCount = MAX_OBSTACLES
        self.previousHead = 0, 0
        self.score = 0
        self.gameOver = False
        self.generateObstacles()
        self.generateRandomFoodLocation()

    
    def gameFinished(self, frame, win: bool):
        if win:
            cv2.putText(frame, 'YOU WIN!!', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 3)
        else:
            cv2.putText(frame, 'GAME OVER!!', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
        cv2.imshow("Camera", frame)
        res = mb.askquestion('Exit Application', 'Play again?')       
        if res == 'yes' :
            self.reset()
        else:
            global playGame
            playGame = False
        

    def updateSnake(self, image, NewHeadLocation):
        self.points.append(NewHeadLocation)
        dist_to_last_head = distance(self.previousHead, NewHeadLocation)
        self.lengths.append(dist_to_last_head)
        self.currentLength += dist_to_last_head
        self.previousHead = NewHeadLocation

        # Reduce current length if more than total length
        if self.currentLength > self.totalLength:
            for len_i, length in enumerate(self.lengths):
                self.currentLength -= length
                self.points.pop(len_i)
                self.lengths.pop(len_i)
                if self.currentLength < self.totalLength:
                    break
        
        # Check if food was eaten
        if checkPointOverlapImage(NewHeadLocation, self.foodLocation, self.foodImageSize):
            self.totalLength += LENGTH_INCREASE_FROM_FOOD
            self.score += 1
            self.generateRandomFoodLocation()
        
        # Display snake, food, and obstacles
        image = self.drawObjects(image)

        # Display score
        cv2.putText(image, 'Score :' + str(self.score), (450, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 203), 2)
        
        # Check collision with obstacles
        for location in self.obstacleLocations:
            if checkPointOverlapImage(NewHeadLocation, location, self.obstacleImageSize):
                self.gameFinished(image, False)
        
        # Check collision with snake body
        check_pts = np.array(self.points[:-6], np.int32)
        check_pts = check_pts.reshape((-1,1,2))
        cv2.polylines(image, [check_pts], False, (0,255,0), 3)
        minimum_dist = cv2.pointPolygonTest(check_pts, NewHeadLocation, True)
        if -1 <= minimum_dist<= 1:
            self.gameFinished(image, False)

        # Check win condition
        if self.score >= WIN_SCORE:
            self.gameFinished(image, True)
        
        return image


def erode(mask, kernel_size=(3,3), iterations=1):
    kernel = np.ones(kernel_size, np.uint8)
    eroded_mask = mask.copy()
    for _ in range(iterations):
        eroded_mask = np.minimum.reduce([
            eroded_mask[i:mask.shape[0]-kernel_size[0]+i+1, j:mask.shape[1]-kernel_size[1]+j+1] 
            for i in range(kernel_size[0]) 
            for j in range(kernel_size[1])])
    return eroded_mask

# Function for dilation
def dilate(mask, iterations=2):
    dilated_mask = mask.copy()
    for _ in range(iterations):
        dilated_mask[1:-1, 1:-1] |= dilated_mask[:-2, 1:-1]
        dilated_mask[1:-1, 1:-1] |= dilated_mask[2:, 1:-1]
        dilated_mask[1:-1, 1:-1] |= dilated_mask[1:-1, :-2]
        dilated_mask[1:-1, 1:-1] |= dilated_mask[1:-1, 2:]
    return dilated_mask

### ONLY USED FOR WELZL

# Structure to represent a 2D point
class Point:
    def __init__(self,X=0,Y=0) -> None:
        self.X=X
        self.Y=Y

# Structure to represent a 2D circle
class Circle:
    def __init__(self,c=Point(),r=0) -> None:        
        self.C=c
        self.R=r

# Function to return the euclidean distance
# between two points
def dist(a: Point, b: Point):
    return math.sqrt(pow(a.X - b.X, 2) + pow(a.Y - b.Y, 2))

# Function to check whether a point lies inside
# or on the boundaries of the circle
def is_inside(c: Circle, p: Point):
    return dist(c.C, p) <= c.R

# The following two functions are used
# To find the equation of the circle when
# three points are given.

# Helper method to get a circle defined by 3 points
def get_circle_center(bx, by, cx, cy):
    B = bx * bx + by * by
    C = cx * cx + cy * cy
    D = bx * cy - by * cx
    return Point((cy * B - by * C) / (2 * D), (bx * C - cx * B) / (2 * D))

# Function to return the smallest circle
# that intersects 2 points
def circle_from1(A: Point, B: Point):
    # Set the center to be the midpoint of A and B
    C = Point((A.X + B.X) / 2.0, (A.Y + B.Y) / 2.0 )
    # Set the radius to be half the distance AB
    return Circle(C, dist(A, B) / 2.0 )

# Function to return a unique circle that
# intersects three points
def circle_from2(A: Point, B: Point, C: Point):
    I = get_circle_center(B.X - A.X, B.Y - A.Y, C.X - A.X, C.Y - A.Y)
    I.X += A.X
    I.Y += A.Y
    return Circle(I, dist(I, A))

# Function to check whether a circle
# encloses the given points
def is_valid_circle(c: Circle, P: list[Point]):
    # Iterating through all the points
    # to check  whether the points
    # lie inside the circle or not
    for p in P:
        if (not is_inside(c, p)):
            return False
    return True

# Function to return the minimum enclosing
# circle for N <= 3
def min_circle_trivial(P: list[Point]):
    assert(len(P) <= 3)
    if not P :
        return Circle() 
    elif (len(P) == 1) :
        return Circle(P[0], 0) 
    elif (len(P) == 2) :
        return circle_from1(P[0], P[1])
    # To check if MEC can be determined
    # by 2 points only
    for i in range(3):
        for j in range(i + 1,3):
            c = circle_from1(P[i], P[j])
            if (is_valid_circle(c, P)):
                return c
    return circle_from2(P[0], P[1], P[2])

# Returns the MEC using Welzl's algorithm
# Takes a set of input points P and a set R
# points on the circle boundary.
# n represents the number of points in P
# that are not yet processed.
def welzl_helper(P: list[Point], R: list[Point], n: int):
    # Base case when all points processed or |R| = 3
    if (n == 0 or len(R) == 3) :
        return min_circle_trivial(R)
    # Pick a random point randomly
    idx = randint(0,n-1)
    p = P[idx]
    # Put the picked point at the end of P
    # since it's more efficient than
    # deleting from the middle of the vector
    P[idx], P[n - 1]=P[n-1],P[idx]
    # Get the MEC circle d from the
    # set of points P - :p
    d = welzl_helper(P, R.copy(), n - 1)
    # If d contains p, return d
    if (is_inside(d, p)):
        return d
    # Otherwise, must be on the boundary of the MEC
    R.append(p)
    # Return the MEC for P - :p and R U :p
    return welzl_helper(P, R.copy(), n - 1)
 
def welzl(P: list[Point]):
    P_copy = P.copy()
    shuffle(P_copy)
    return welzl_helper(P_copy, [], len(P_copy))





playGame = True
game = SnakeGame("apple.jpg", "hurdle1small.jpg")
cap = cv2.VideoCapture(0)
cap.set(3, 720)
cap.set(4, 720)


while playGame:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    img = cv2.GaussianBlur(frame, (11, 11), 2)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = ((np.logical_and(np.all(img >= GREEN_LOWER_THRESHOLD, axis = -1), np.all(img <= GREEN_UPPER_THRESHOLD, axis = -1))).astype(np.uint8))*255
    mask = erode(mask)
    mask = dilate(mask)

    # find contours
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    if len(contours) > 0:
        ball_contour = max(contours, key=cv2.contourArea)
        # Assuming ball_contour is the largest contour obtained from the binary mask image

        # Convert the contour to a list of points
        points = [Point(x[0][0], x[0][1]) for x in ball_contour]

        # Find the minimum enclosing circle using Welzl's algorithm
        min_enclosing_circle = welzl(points)
        center = (int(min_enclosing_circle.C.X), int(min_enclosing_circle.C.Y))
        radius = int(min_enclosing_circle.R)

        if radius > 10:
            frame = game.updateSnake(frame, center)
        else:
            frame = game.drawObjects(frame)
    else:
        frame = game.drawObjects(frame)
    cv2.imshow("Camera", frame)
    cv2.imshow("Mask", mask)
    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()

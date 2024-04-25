import cv2
import imutils
from collections import deque
import numpy as np
import time

# Initialize game variables
score = 0
max_score = 20
snake_length = 5
snake_body = deque(maxlen=200)
apple_position = None
hurdles = []

# Define the number of hurdles
num_hurdles = 5

# Initialize flag for game state
game_over = False
game_won = False

# Define color ranges for the snake's movement
green_lower = (29, 86, 18)
green_upper = (93, 255, 255)

# Distance function
def dist(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)

# Function to generate random apple and hurdle positions
def generate_positions(frame):
    global apple_position
    global hurdles
    if apple_position is None:
        # Generate random apple position
        apple_x = np.random.randint(30, frame.shape[1] - 30)
        apple_y = np.random.randint(30, frame.shape[0] - 30)
        apple_position = (apple_x, apple_y)
    
    if not hurdles:
        # Generate random hurdle positions
        hurdles = [(np.random.randint(30, frame.shape[1] - 30), np.random.randint(30, frame.shape[0] - 30)) for _ in range(num_hurdles)]

# Function to check if the snake's head collides with any hurdles
def check_collision(snake_head, hurdles):
    for hurdle in hurdles:
        if dist(snake_head, hurdle) < 10:
            return True
    return False

# Initialize video capture
cap = cv2.VideoCapture(0)

# Snake game loop
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=600)
    img = cv2.GaussianBlur(frame, (11, 11), 0)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Generate random apple and hurdle positions if needed
    generate_positions(frame)
    
    # Draw apple on the frame
    cv2.circle(frame, apple_position, 10, (0, 0, 255), -1)
    
    # Draw hurdles on the frame
    for hurdle in hurdles:
        cv2.circle(frame, hurdle, 10, (0, 255, 255), -1)
    
    # Mask out the green color and preprocess the mask
    mask = cv2.inRange(img, green_lower, green_upper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    # Find contours in the mask
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    if cnts:
        ball_contour = max(cnts, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(ball_contour)
        
        # Calculate center of the contour (snake head)
        M = cv2.moments(ball_contour)
        snake_head = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        
        if radius > 10:
            cv2.circle(frame, snake_head, 5, (0, 0, 255), -1)
            
            # Add snake head position to snake body
            snake_body.appendleft(snake_head)
            
            # Check if the snake's head collides with the apple
            if dist(apple_position, snake_head) < 10:
                score += 1
                
                # Generate a new apple position
                apple_position = None
                
                # Increase snake length
                snake_length += 5
                
                # Check if the player has won the game
                if score == max_score:
                    game_won = True
            
            # Check if the snake's head collides with any hurdles
            if check_collision(snake_head, hurdles):
                game_over = True
            
            # Draw the snake body
            for i in range(1, len(snake_body)):
                if snake_body[i - 1] is None or snake_body[i] is None:
                    continue
                r, g, b = np.random.randint(0, 255, 3)
                r, g, b = int(r), int(g), int(b)
                thickness = int(len(snake_body) / 10) + 2
                cv2.line(frame, snake_body[i], snake_body[i - 1], (r, g, b), thickness)
    
    # Display the score on the frame
    cv2.putText(frame, f'Score: {score}', (450, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 203), 2)
    
    # Check game win condition and display appropriate messages
    if game_won:
        cv2.putText(frame, 'YOU WIN!!', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 3)
        cv2.imshow('Live Feed', frame)
        cv2.waitKey(3000)  # Wait for 3 seconds before ending the game
        break
    
    # Check game loss condition and display appropriate messages
    if game_over:
        cv2.putText(frame, 'YOU LOST!!', (100, 250), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
        cv2.imshow('Live Feed', frame)
        cv2.waitKey(3000)  # Wait for 3 seconds before ending the game
        break
    
    # Display the frames
    cv2.imshow('Live Feed', frame)
    cv2.imshow('Mask', mask)
    
    # Break the loop if the user presses the ESC key
    if cv2.waitKey(1) == 27:
        break

# Cleanup
cv2.destroyAllWindows()
cap.release()

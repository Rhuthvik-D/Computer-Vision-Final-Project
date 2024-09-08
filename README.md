
# Snake Game with OpenCV and Welzl's Algorithm

## Overview
This project implements a Snake game using Python, OpenCV, and computer vision techniques. The game incorporates object detection via the webcam, which tracks a green object to control the movement of the snake. The goal of the game is to eat the food while avoiding obstacles. Additionally, Welzl's algorithm is applied to find the minimum enclosing circle around detected objects for more accurate tracking.

## Features
- **Webcam-Controlled Snake**: Move the snake using a green object (like a ball or your hand with a green glove).
- **Dynamic Obstacles**: The game generates random obstacles that you must avoid.
- **Real-Time Food Detection**: The snake can eat food, which increases its length.
- **Game Over and Win Conditions**: The game finishes if you collide with your body or an obstacle, or if you reach the winning score.

## Prerequisites
To run this project, you'll need the following Python packages:
- OpenCV (`opencv-python`)
- Imutils (`imutils`)
- Numpy (`numpy`)
- Tkinter (`tkinter`)

You can install the necessary packages using pip:
```bash
pip install opencv-python imutils numpy tkinter
```

## How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/snake-game-opencv.git
   ```
2. Navigate to the project directory:
   ```bash
   cd snake-game-opencv
   ```
3. Ensure your webcam is connected and working.
4. Run the main game script:
   ```bash
   python snake_game.py
   ```

## Game Pipeline
1. **Webcam Initialization**: The game starts by capturing frames from the webcam.
2. **Image Preprocessing**:
   - The frames are blurred and converted from BGR to HSV format.
   - A binary mask is created to detect the green object (used to control the snake).
   - Erosion and dilation are applied to remove noise and refine the object detection.
3. **Object Detection**:
   - Contours are detected from the binary mask.
   - The largest contour is assumed to be the green object.
4. **Minimum Enclosing Circle**:
   - Welzl's algorithm is applied to find the minimum enclosing circle around the detected contour, allowing accurate detection of the object’s center and size.
5. **Snake Movement**:
   - The snake's head follows the center of the detected green object.
   - The snake grows when it eats food and avoids dynamically generated obstacles.
6. **Collision Detection**:
   - The game checks for collisions between the snake's head and its body, as well as obstacles.
7. **Game Over/Win Conditions**:
   - The game ends if the snake collides with its body or an obstacle.
   - The player wins if they reach the predefined score (`WIN_SCORE`).

## Key Concepts Used

### 1. **OpenCV for Real-Time Processing**
   - Capturing video feed from the webcam (`cv2.VideoCapture`).
   - Applying Gaussian blur to smooth the image.
   - Thresholding the image in HSV format to detect the green object using color range filtering.
   - Contour detection for tracking the position of the green object.

### 2. **Welzl's Algorithm for Minimum Enclosing Circle**
   - Welzl's algorithm is used to find the smallest enclosing circle around the detected contour points.
   - This ensures precise tracking of the object’s center and radius.

### 3. **Erosion and Dilation**
   - **Erosion**: Reduces noise in the binary mask by shrinking the regions, effectively removing small unwanted areas.
   - **Dilation**: Enlarges the regions in the binary mask to recover the object's size after erosion.

### 4. **Game Logic (Snake Movement and Collision)**
   - Implemented using the `SnakeGame` class:
     - Snake movement is controlled by the green object.
     - The snake's length increases when food is eaten.
     - Game over occurs when the snake collides with its own body or with obstacles.
     - Win condition is achieved by reaching the `WIN_SCORE`.

## Code Breakdown

### 1. **Distance Calculation**
   The `distance()` function calculates the Euclidean distance between two points, which is crucial for checking distances between the snake, food, and obstacles.

### 2. **Check Point Overlap**
   The `checkPointOverlapImage()` function checks if a point (snake head) overlaps with an image (food or obstacle) on the game screen.

### 3. **Welzl's Algorithm**
   - Welzl’s algorithm is implemented to compute the minimum enclosing circle of a set of points (the contour detected from the green object).
   - It works recursively and finds the smallest possible circle that encloses the given points.

### 4. **SnakeGame Class**
   The main game logic resides in the `SnakeGame` class, which controls:
   - **Snake Movement**: The snake moves based on the position of the green object detected.
   - **Food and Obstacle Generation**: Random positions are generated for the food and obstacles, ensuring no overlap.
   - **Collision Detection**: Handles collision with obstacles and the snake's body.
   - **Game Reset**: Resets the game after winning or losing, asking the user if they want to play again.
   
### 5. **Erosion and Dilation**
   - The `erode()` function applies erosion to remove small noisy regions from the mask.
   - The `dilate()` function expands regions to ensure the object is correctly represented after erosion.

### 6. **Game Over and Win Logic**
   - The game checks for collisions between the snake and obstacles using `checkPointOverlapImage()`.
   - It also checks if the snake eats food by comparing the snake head’s position with the food’s position.
   - When the snake reaches the winning score (`WIN_SCORE`), the game declares a victory.

## Game Controls
- **Movement**: Use a green object (such as a ball or a glove) to control the snake's movement via webcam.
- **Goal**: Move the snake to eat the food (apple) and avoid obstacles.
- **Score**: Eating food increases the score and lengthens the snake.
- **Win Condition**: Reach the predefined score (default is 10) to win.
- **Lose Condition**: The game ends if the snake collides with its body or an obstacle.

## How to Win
- Keep eating the food to increase your score and lengthen the snake.
- Avoid obstacles and the snake's body as it grows longer.
- Reach the winning score (`WIN_SCORE`, default is 10) to win.


# findFlag.py

from detecto.core import Model
import cv2 #Used for loading the image into memory

# First, let's load our trained model from the Training section
# We need to specify the label which we want to find (the same one from Classification and Training)
model = Model.load('model.pth', ['aboriginal_flag'])

# Now, let's load a sample image into memory
# Change the file name below if you want to test other potential samples
image = cv2.imread("samples/sample4.jpg")

# model.predict() is the method we call with our image as an argument
# to try find our desired object in the sample image using our pre-trained model.
# It will do a bit of processing and then spit back some numbers.
# The numbers define what it thinks the bounding boxes are of potential matches.
# And the probability that the bounding box is recognizing the object (flag).
labels, boxes, scores = model.predict(image)

# Below we are just printing the results, predict() will
# give back a couple of arrays that represent the bounding box coordinates and
# probability that the model believes that the box is a match
# The coordinates are (xMin, yMin, xMax, yMax)
# Using this data, you could just open the original image in an image editor
# and draw a box around the printed coordinates
print(labels, boxes, scores)

# WARNING: You don't have to understand this part, I barely do.
# All this code does is draw rectangles around the model predictions above
# and outputs to the display for your viewing pleasure.
for idx, s in enumerate(scores):
    if s > 0.1: # This line decides what probabilities we should outline
        rect = boxes[idx]
        start_point = (rect[0].int(), rect[1].int())
        end_point = (rect[2].int(), rect[3].int())
        cv2.rectangle(image, start_point, end_point, (0, 0, 255), 2)

cv2.imshow("Image" + str(idx), image)
cv2.waitKey(0)

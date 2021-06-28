Number Prediction From Hand Gestures


Our aim is to predict the number shown through hand gestures. For this, we trained the hand image model with CNN and used a sliding window on the input image which detects the location and finger number simultaneously . This also helps in identifying multiple hands along with the digits they represent and this was the main purpose of the sliding window . To do this type of multiple detection , we have decided a threshold for the score and then we draw a box around the detected hands and also provide a list of detected numbers.
We have also included a special feature which allows the user to open their camera and capture the image and then we predict the hand and number detected by it.

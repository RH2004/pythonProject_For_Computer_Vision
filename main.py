import cv2
#print(cv2.__version__)

#----------------------------------------------------------------------------------------------------------------
'''
# Milestone 1 : Setting Up Environment
# Ok now we will accesss the default camera of the pc
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read() # we will be capturing frame by frame the image
    cv2.imshow('frame', frame) # Display the frame
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()  # Release the capture
cv2.destroyAllWindows()  # destroy the window
'''
#----------------------------------------------------------------------------------------------------------------

'''
# Milestone 2 : Face Detection
# Load the face detection model
facecascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Capture video from webcam (index 0)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the webcam
    ret, frame = cap.read()

    # Check if frame is read successfully
    if not ret:
        print("Error: Failed to capture frame from webcam")
        break

    # Convert frame to grayscale (better for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = facecascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw a green rectangle around each detected face
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the frame with detected faces
    cv2.imshow('frame', frame)

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
'''
#----------------------------------------------------------------------------------------------------------------

''''
# Milestone 3 : Face Recognition with a Basic Classifier
import cv2

# Load the pre-trained Haar Cascade classifiers for face and animal face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cat_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
dog_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Access the default camera (usually the first camera connected)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Capture frame-by-frame

    # Convert frame to grayscale (required for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    cats = cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    dogs = dog_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangles around detected faces and animals
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # BGR color format

    for (x, y, w, h) in cats:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # BGR color format

    for (x, y, w, h) in dogs:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # BGR color format

    # Display the frame with detected faces and animals
    cv2.imshow('REDAs Face Detection Software', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()  # Release the capture
cv2.destroyAllWindows()  # Close all OpenCV windows
'''
#----------------------------------------------------------------------------------------------------------------
# Milestone 4 : adding labels
import cv2

# Load the pre-trained Haar Cascade classifiers for face and animal face detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cat_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface.xml')
dog_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')

# Access the default camera (usually the first camera connected)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()  # Capture frame-by-frame

    # Convert frame to grayscale (required for face detection)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    cats = cat_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    dogs = dog_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Label detected faces as human or animal
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # BGR color format
        cv2.putText(frame, 'Human', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    for (x, y, w, h) in cats:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # BGR color format
        cv2.putText(frame, 'Cat', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for (x, y, w, h) in dogs:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # BGR color format
        cv2.putText(frame, 'Dog', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Display the frame with detected faces and animals
    cv2.imshow('REDAs Face Detection Software', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()

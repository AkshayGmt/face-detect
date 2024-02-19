import cv2

# Load the Haar cascade classifiers for face and eye detection
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
eye_closed_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')

# Initialize the video capture device
cap = cv2.VideoCapture(0)
#Variable store execution state
first_read = True
while True:
    # Read a frame from the video stream
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around the detected faces and calculate the centers
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        center_x = x + w // 2
        center_y = y + h // 2
        cv2.circle(frame, (center_x, center_y), 2, (0, 0, 255), -1)

        # Detect eyes in the grayscale face region
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # Check if the eyes are closed
        for (ex, ey, ew, eh) in eyes:
            eye_roi_gray = roi_gray[ey:ey+eh, ex:ex+ew]
            eye_roi_color = roi_color[ey:ey+eh, ex:ex+ew]
            eye_closed = eye_closed_cascade.detectMultiScale(eye_roi_gray)
            if len(eye_closed) == 0:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)

                if(len(eyes)>=2):
    				#Check if program is running for detection
                    if(first_read):
                        cv2.putText(frame,
                        "Eye detected press s to begin",
                        (70,70),
                        cv2.FONT_HERSHEY_PLAIN, 3,
                        (0,255,0),2)
                    else:
                        cv2.putText(frame,
                        "Eyes open!", (70,70),
                        cv2.FONT_HERSHEY_PLAIN, 2,
                        (255,255,255),2)
                else:
                    if(first_read):
                        #To ensure if the eyes are present before starting
                        cv2.putText(frame,
                        "No eyes detected", (70,70),
                        cv2.FONT_HERSHEY_PLAIN, 3,
                        (0,0,255),2)
                    else:
                        #This will print on console and restart the algorithm
                        print("Blink detected--------------")
                        cv2.waitKey(3000)
                        first_read=True
                
        

	#Controlling the algorithm with keys
	

    # Display the frame
    cv2.imshow('Frame', frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()

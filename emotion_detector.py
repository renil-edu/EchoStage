# import cv2
# from deepface import DeepFace

# # Load the pre-trained emotion detection model
# model = DeepFace.build_model("Emotion")

# # Define emotion labels
# emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# # Load Haar cascade classifier
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# # Start webcam
# cap = cv2.VideoCapture(0)

# print("Press 'q' to exit the window")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Convert each frame to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces
#     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1,
#                                           minNeighbors=5, minSize=(30, 30))

#     for (x, y, w, h) in faces:
#         face_roi = gray[y:y+h, x:x+w]
#         resized_face = cv2.resize(face_roi, (48, 48))
#         normalized_face = resized_face / 255.0
#         reshaped_face = normalized_face.reshape(1, 48, 48, 1)

#         # Predict emotion
#         preds = model.predict(reshaped_face)[0]
#         emotion = emotion_labels[preds.argmax()]

#         # Draw detection box and label
#         cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#         cv2.putText(frame, emotion, (x, y - 10),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

#     cv2.imshow('Emotion Recognition', frame)

#     # Quit on 'q'
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




import cv2
from deepface import DeepFace

# Initialize webcam capture
cap = cv2.VideoCapture(0)

print("Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Analyze emotions in the current frame
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    # Extract dominant emotion of the first detected face
    emotion = result[0]['dominant_emotion']

    # Display detected emotion on the frame
    cv2.putText(frame, emotion, (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the video frame with emotion label
    cv2.imshow('Emotion Detector', frame)

    # Exit loop if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close window
cap.release()
cv2.destroyAllWindows()

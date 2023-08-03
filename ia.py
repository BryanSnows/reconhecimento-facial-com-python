import cv2
import os

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

cap = cv2.VideoCapture(0)
detected_faces = []

while True:
    ret, frame = cap.read()

    if ret:
        faces = detect_faces(frame)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            detected_faces.append((x, y, w, h))

           
            face_image = frame[y:y+h, x:x+w]
            if not os.path.exists('faces'):
                os.makedirs('faces')
            cv2.imwrite(f'faces/face_{len(detected_faces)}.png', face_image)

        cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
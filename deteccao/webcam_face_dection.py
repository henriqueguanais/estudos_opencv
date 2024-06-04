import cv2

face_detector = cv2.CascadeClassifier('deteccao\Cascades\haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

while True:
    _, frame = video_capture.read()

    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = face_detector.detectMultiScale(image_gray, minSize=(130, 130), minNeighbors=8, scaleFactor=1.02)

    if len(detections) >= 1:
        print(f"Rostos detectados: {len(detections)}")
    else:
        print(f"Rostos detectados: 0")

    for x, y, w, h in detections:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
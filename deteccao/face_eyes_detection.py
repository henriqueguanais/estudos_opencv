import cv2

image = cv2.imread('deteccao/Images/people1.jpg')
image = cv2.resize(image, (800, 600))
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

eye_detection = cv2.CascadeClassifier('deteccao/Cascades\haarcascade_eye.xml')
facial_detection = cv2.CascadeClassifier('deteccao/Cascades\haarcascade_frontalface_default.xml')
f_detections = facial_detection.detectMultiScale(gray_image, scaleFactor=1.08) #scaleFactor muda a escala da imagem

print("Caras:\n")
for x, y, w, h in f_detections:
    print(w, y)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2) #(imagem, (coordenada x,y), (coordenas + width and height), (cor bgr), grossura)

print("Olhos:\n")
e_detections = eye_detection.detectMultiScale(gray_image, scaleFactor=1.09, minNeighbors=9)
for x, y, w, h in e_detections:
    print(w, h)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

cv2.imshow('Teste', image)


cv2.waitKey(0)
cv2.destroyAllWindows()
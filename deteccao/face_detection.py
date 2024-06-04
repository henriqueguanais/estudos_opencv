import cv2

image = cv2.imread('deteccao/Images/people2.jpg')
image = cv2.resize(image, (600, 400))
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

facial_detection = cv2.CascadeClassifier('deteccao/Cascades\haarcascade_frontalface_default.xml')#minNeighbors detecta o melhor candidato entre alguns proximos
detections = facial_detection.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=3,
                                               minSize=(20, 20), maxSize=(89, 85)) # scaleFactor especifica o quanto a imagem de entrada é reduzida em cada escala de imagem.
# quanto maior minNeighbors, mais preciso será a detecção, mas pode ser mais lento
# quanto menor scaleFactor, mais preciso será a detecção, mas pode ser mais lento

for x, y, w, h in detections:
    print(w, h)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
cv2.imshow('Teste', image)

cv2.waitKey(0)
cv2.destroyAllWindows()
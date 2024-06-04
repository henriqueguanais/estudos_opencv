import cv2

clock_image = cv2.imread('deteccao/Images/clock.jpg')
gray_clock = cv2.cvtColor(clock_image, cv2.COLOR_BGR2GRAY)
car_image = cv2.imread('deteccao\Images\car.jpg')
gray_car = cv2.cvtColor(car_image, cv2.COLOR_BGR2GRAY)

clock_detector = cv2.CascadeClassifier('deteccao/Cascades/clocks.xml')
c_detections = clock_detector.detectMultiScale(gray_clock, scaleFactor=1.01, minNeighbors=8)
car_detector = cv2.CascadeClassifier('deteccao/Cascades\cars.xml')
cr_detections = car_detector.detectMultiScale(gray_car, scaleFactor=1.05, minNeighbors=1,
                                              maxSize=(70, 70), minSize=(30, 30))

print(f"Número de relógios: {len(c_detections)}")
for x, y, w, h in c_detections:
    cv2.rectangle(clock_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.imshow('Relógios', clock_image)

print(f"Número de carros: {len(cr_detections)}")
for x, y, w, h in cr_detections:
    print(w, h)
    cv2.rectangle(car_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
cv2.imshow('Carros', car_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
import cv2 
from PIL import Image
import numpy as np
import os

lbph_face_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_face_classifier.read("reconhecimento_facial\classifier\lbph_classifier.yml")

image_test = 'reconhecimento_facial\\Datasets\\yalefaces\\test\\subject05.surprised.gif'
image = Image.open(image_test).convert('L')
image_np = np.array(image, 'uint8')

previsao = lbph_face_classifier.predict(image_np)
print(previsao)

saida_esperada = int(os.path.split(image_test)[1].split('.')[0].replace('subject', ''))

cv2.putText(image_np, f"Pred: {previsao[0]}", (10, 30), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
cv2.putText(image_np, f"Exp: {saida_esperada}", (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0))
cv2.imshow("teste", image_np)

cv2.waitKey()
cv2.destroyAllWindows()
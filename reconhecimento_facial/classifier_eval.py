import cv2
import os
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn
import matplotlib.pyplot as plt

lbph_classifier = cv2.face.LBPHFaceRecognizer_create()
lbph_classifier.read("reconhecimento_facial\classifier\lbph_classifier3.yml")

paths = [os.path.join("reconhecimento_facial\\datasets\\yalefaces\\test", f) for f in os.listdir("reconhecimento_facial\\datasets\\yalefaces\\test")]
previsoes = []
saidas_esperadas = []
for path in paths:
    # print(path)
    imagem = Image.open(path).convert('L')
    imagem_np = np.array(imagem, 'uint8')
    previsao,_ = lbph_classifier.predict(imagem_np)
    # print(previsao)
    # pega so o numero do caminho
    saida_esperada = int(os.path.split(path)[1].split('.')[0].replace('subject', ''))
    # print(saida_esperada)

    previsoes.append(previsao)
    saidas_esperadas.append(saida_esperada)

# print(type(previsoes), type(saidas_esperadas))
# converter essas listas para o formato numpy
previsoes = np.array(previsoes)
saidas_esperadas = np.array(saidas_esperadas)
# print(type(previsoes), type(saidas_esperadas))
# print(previsoes, '\n', saidas_esperadas)

# Acuracia
print(accuracy_score(saidas_esperadas, previsoes))
# matriz de confusao
cm = confusion_matrix(saidas_esperadas, previsoes)
# print(cm)

seaborn.heatmap(cm, annot=True, fmt="d")
plt.show()
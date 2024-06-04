from PIL import Image
import cv2
import numpy as np
import os

def get_image_data():
    paths = [os.path.join('reconhecimento_facial\\Datasets\\yalefaces\\train', f) for f in os.listdir('reconhecimento_facial\\Datasets\\yalefaces\\train')] #coloca o nome dos arquivos + o caminho
    faces = []
    ids = []
    for path in paths:
        image = Image.open(path).convert('L') # converte a imagem em escala de cinza
        image_np = np.array(image, 'uint8') #cria uma matriz so com numeros inteiros
        id = int(os.path.split(path)[1].split('.')[0].replace('subject', '')) #pega somente o numero do id da string path
        #print(id)
        ids.append(id)
        faces.append(image_np)
    
    return np.array(ids), faces

ids, faces = get_image_data()

# print(os.getcwd())
lbph_classifier = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=8, grid_x=8, grid_y=8)  #cria o classificador de faces
lbph_classifier.train(faces, ids)  #treina o classificador
lbph_classifier.write('reconhecimento_facial\\classifier\\lbph_classifier4.yml')   #cria o arquivo .yml
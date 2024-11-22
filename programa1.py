import cv2
import numpy as np
import matplotlib.pyplot as plt

#leer imagen
imagen = cv2.imread('C:/Users/TereH/OneDrive/Escritorio/IA/imagen.jpg')
gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
bordes = cv2.Canny(gris, 100,200)

cv2.imshow('imagen Original', imagen)
cv2.imshow('imagen color gris', gris)
cv2.imshow('imagen gris bordes', bordes)

cv2.waitKey(0)

cv2.destroyAllWindows()




#muestra la imagen en unq ventana
#if imagen is None:
 #   print('No se pudo abrir la imagen.')
#else:
#    cv2.imshow('Imagen', imagen)
 #   cv2.waitKey(0)
 #   cv2.destroyAllWindows()
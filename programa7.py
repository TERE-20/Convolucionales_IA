import numpy as np
import cv2
import imutils

def ordenar_puntos(puntos):
    n_puntos = np.concatenate([puntos[0], puntos[1], puntos[2], puntos[3]]).tolist()
    y_order = sorted(n_puntos, key=lambda n_puntos: n_puntos[1])
    
    x1_order = y_order[:2]
    x1_order = sorted(x1_order, key=lambda x1_order: x1_order[0])
    
    x2_order = y_order[2:4]
    x2_order = sorted(x2_order, key=lambda x2_order: x2_order[0])
    
    return [x1_order[0], x1_order[1], x2_order[0], x2_order[1]]

def roi(image, ancho, alto):
    imagen_alineada = None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    cnts = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:1]
    for c in cnts:
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) == 4:
            puntos = ordenar_puntos(approx)
            pts1 = np.float32(puntos)
            pts2 = np.float32([[0, 0], [ancho, 0], [0, alto], [ancho, alto]])
            M = cv2.getPerspectiveTransform(pts1, pts2)
            imagen_alineada = cv2.warpPerspective(image, M, (ancho, alto))
    return imagen_alineada

def detectar_figuras(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canny = cv2.Canny(gray, 10, 150)
    canny = cv2.dilate(canny, None, iterations=1)
    canny = cv2.erode(canny, None, iterations=1)
    cnts = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for c in cnts:
        epsilon = 0.01 * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, epsilon, True)
        x, y, w, h = cv2.boundingRect(approx)

        if len(approx) == 3:
            cv2.putText(image, 'Triangulo', (x, y - 5), 1, 1, (0, 255, 0), 1)
        elif len(approx) == 4:
            aspect_ratio = float(w) / h
            if 0.95 <= aspect_ratio <= 1.05:
                cv2.putText(image, 'Cuadrado', (x, y - 5), 1, 1, (0, 255, 0), 1)
            else:
                cv2.putText(image, 'Rectangulo', (x, y - 5), 1, 1, (0, 255, 0), 1)
        elif len(approx) == 5:
            cv2.putText(image, 'Pentagono', (x, y - 5), 1, 1, (0, 255, 0), 1)
        elif len(approx) == 6:
            cv2.putText(image, 'Hexagono', (x, y - 5), 1, 1, (0, 255, 0), 1)
        elif len(approx) > 10:
            cv2.putText(image, 'Circulo', (x, y - 5), 1, 1, (0, 255, 0), 1)

        cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("No se pudo acceder a la cámara.")
        break

    imagen_A4 = roi(frame, ancho=720, alto=509)
    if imagen_A4 is not None:
        detectar_figuras(imagen_A4)
        cv2.imshow('Detección de Figuras', imagen_A4)

    cv2.imshow('Vista Original', frame)

    if cv2.waitKey(1) & 0xFF == 'q': 
        break

cap.release()
cv2.destroyAllWindows()

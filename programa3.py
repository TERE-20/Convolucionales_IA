import cv2

captura = cv2.VideoCapture(0)

ret, frame = captura.read()

if  ret:
    cv2.imwrite('C:/Users/TereH/OneDrive/Escritorio/IA/captura.jpg', frame)
        
captura.release()
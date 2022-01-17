import cv2

faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
imagen = cv2.imread('oficina.png')
imagen2 = cv2.imread('personas.jpg')
imGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
imGris2 = cv2.cvtColor(imagen2, cv2.COLOR_BGR2GRAY)

#se hace la deteccion de las dos imagenes de personas 
rostros = faceClassif.detectMultiScale(imGris,
    scaleFactor = 1.1,
    minNeighbors = 5,
    minSize = (30,30),
    maxSize = (200,200))

rostros2 = faceClassif.detectMultiScale(imGris2,
    scaleFactor = 1.1,
    minNeighbors = 8,
    minSize = (30,30),
    maxSize = (200,200))

for (x,y,w,h) in rostros:
    cv2.rectangle(imagen,(x,y),(x+w,y+h),(0,255,0),2)

for (x,y,w,h) in rostros2:
    cv2.rectangle(imagen2,(x,y),(x+w,y+h),(0,0,255),2)

cv2.imshow('imagen', imagen)
cv2.imshow('Rostros', imagen2)
cv2.waitKey(0)
cv2.destroyAllWindows()

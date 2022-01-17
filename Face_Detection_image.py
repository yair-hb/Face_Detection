import cv2

faceClassif = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
imagen = cv2.imread('oficina.png')
imGris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

rostros = faceClassif.detectMultiScale(imGris,
    scaleFactor = 1.1,
    minNeighbors = 5,
    minSize = (30,30),
    maxSize = (200,200))
for (x,y,w,h) in rostros:
    cv2.rectangle(imagen,(x,y),(x+w,y+h),(0,255,0),2)
    
cv2.imshow('imagen', imagen)
cv2.waitKey(0)
cv2.destroyAllWindows()

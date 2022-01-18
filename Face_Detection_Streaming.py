import cv2

captura = cv2.VideoCapture(0)
faceClassif = cv2.CascadeClassifier('haarcasacde_forntalface_default.xml')

while True:
    ret,frame = captura.read()
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rostros = faceClassif.detectMultiScale(gris,1.3,5)

    for (x,y,w,h) in rostros:
        cv2.rectangle(frame,(x,y), (x+w,y+h),(0,255,0),2)

    cv2.imshow('ROSTROS',frame)

    if cv2.waitKey(1) & 0xFF == ord('s'):
        break
captura.release()
cv2.destroyAllWindows()
import cv2
import numpy as np

url = "http://10.22.155.124:4747/video" # IP de DroidCam cambiarla 

cap = cv2.VideoCapture(url)

while True:

    ret, frame = cap.read()

    if not ret:
        print("No se pudo conectar")
        break

    frame = cv2.resize(frame, (640, 480))

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    # ROSA = ZONA A

    rosa_bajo = np.array([140, 50, 50])
    rosa_alto = np.array([170, 255, 255])

    mascara_rosa = cv2.inRange(hsv, rosa_bajo, rosa_alto)

    # NARANJA = ZONA B

    naranja_bajo = np.array([5, 100, 100])
    naranja_alto = np.array([20, 255, 255])

    mascara_naranja = cv2.inRange(hsv, naranja_bajo, naranja_alto)

    pixeles_rosa = cv2.countNonZero(mascara_rosa)
    pixeles_naranja = cv2.countNonZero(mascara_naranja)

    if pixeles_rosa > 5000:

        print("A")

        cv2.putText(
            frame,
            "Zona A",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 255),
            3
        )

    elif pixeles_naranja > 5000:

        print("B")

        cv2.putText(
            frame,
            "Zona B",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 140, 255),
            3
        )

    cv2.imshow("Detector", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

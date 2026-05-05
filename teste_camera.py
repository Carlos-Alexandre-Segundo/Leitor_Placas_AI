import cv2

for i in range(10):
    cap = cv2.VideoCapture(i)

    if cap.isOpened():
        print(f"Câmera encontrada no índice {i}")

        ret, frame = cap.read()

        if ret:
            cv2.imshow(f"Camera {i}", frame)
            cv2.waitKey(3000)

        cap.release()
    else:
        print(f"Nada no índice {i}")

cv2.destroyAllWindows()
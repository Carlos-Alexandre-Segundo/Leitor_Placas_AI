import cv2

backends = [
    ("PADRAO", cv2.CAP_ANY),
    ("DSHOW", cv2.CAP_DSHOW),
    ("MSMF", cv2.CAP_MSMF),
]

for nome, backend in backends:
    for i in range(10):
        print(f"Testando {nome} índice {i}...")

        cap = cv2.VideoCapture(i, backend)

        if cap.isOpened():
            print(f"Câmera encontrada: backend={nome}, índice={i}")

            ret, frame = cap.read()

            if ret:
                cv2.imshow(f"{nome} - Camera {i}", frame)
                cv2.waitKey(3000)

            cap.release()
            cv2.destroyAllWindows()
            exit()

        cap.release()

print("Nenhuma câmera encontrada.")
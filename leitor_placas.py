import cv2
from ultralytics import YOLO
import requests
import re

# ================== CONFIGURAÇÕES ==================

API_BASE_URL = "SUA_API_DE_BASE"
API_ENTRY = f"{API_BASE_URL}SUA_URL"

PLATE_RECOGNIZER_TOKEN = "SEU_TOKEN_PESSOAL"

modelo_yolo = YOLO(
    'SEU_CAMINHO_PARA_O_YOLO'
)

BOTAO_CAPTURAR = {
    "x1": 20,
    "y1": 420,
    "x2": 180,
    "y2": 465
}

clicou_capturar = False


# ================== MOUSE ==================

def clique_mouse(event, x, y, flags, param):
    global clicou_capturar

    if event == cv2.EVENT_LBUTTONDOWN:
        if (
            BOTAO_CAPTURAR["x1"] <= x <= BOTAO_CAPTURAR["x2"] and
            BOTAO_CAPTURAR["y1"] <= y <= BOTAO_CAPTURAR["y2"]
        ):
            clicou_capturar = True


# ================== FUNÇÕES ==================

def desenhar_botao(imagem):
    x1 = BOTAO_CAPTURAR["x1"]
    y1 = BOTAO_CAPTURAR["y1"]
    x2 = BOTAO_CAPTURAR["x2"]
    y2 = BOTAO_CAPTURAR["y2"]

    cv2.rectangle(imagem, (x1, y1), (x2, y2), (255, 255, 255), -1)
    cv2.rectangle(imagem, (x1, y1), (x2, y2), (0, 0, 0), 2)

    cv2.putText(
        imagem,
        "CAPTURAR",
        (x1 + 20, y1 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2
    )


def detectar_placa(imagem):
    resultados = modelo_yolo(imagem, verbose=False)
    melhor_recorte = None
    melhor_confianca = 0
    imagem_original = imagem

    for r in resultados:
        caixas = r.boxes

        if caixas is None:
            continue

        for caixa in caixas:
            classe_id = int(caixa.cls[0])
            nome_classe = modelo_yolo.names[classe_id]
            confianca = float(caixa.conf[0])

            x1, y1, x2, y2 = map(int, caixa.xyxy[0])

            if nome_classe in ['old_plate', 'mercosul_plate'] and confianca >= 0.70:
                cv2.rectangle(imagem, (x1, y1), (x2, y2), (0, 255, 0), 2)

                texto = f"{nome_classe} {confianca:.2f}"
                cv2.putText(
                    imagem,
                    texto,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                if confianca > melhor_confianca:
                    recorte = imagem_original[y1:y2, x1:x2]

                    if recorte is not None and recorte.size > 0:
                        melhor_recorte = recorte.copy()
                        melhor_confianca = confianca

    return melhor_recorte

def aplicar_filtro_placa(recorte):
    # aumenta a imagem para melhorar leitura
    recorte = cv2.resize(recorte, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # converte para cinza
    gray = cv2.cvtColor(recorte, cv2.COLOR_BGR2GRAY)

    # reduz ruído
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # preto e branco com contraste
    _, preto_branco = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return preto_branco

def placa_valida(placa):
    padrao_antigo = r'^[A-Z]{3}[0-9]{4}$'
    padrao_mercosul = r'^[A-Z]{3}[0-9][A-Z][0-9]{2}$'

    return (
        re.match(padrao_antigo, placa) is not None or
        re.match(padrao_mercosul, placa) is not None
    )


def extrair_texto_placa(recorte):
    if recorte is None or recorte.size == 0:
        return None

    caminho_temp = "placa_temp.jpg"
    recorte_filtrado = aplicar_filtro_placa(recorte)
    cv2.imwrite(caminho_temp, recorte_filtrado)

    with open(caminho_temp, 'rb') as imagem:
        response = requests.post(
            'ACESS_PLATE_RECOGNIZER',
            files={'upload': imagem},
            headers={
                'Authorization': f'Token {PLATE_RECOGNIZER_TOKEN}'
            },
            data={
                'regions': 'br'
            },
            timeout=10
        )

    if response.status_code not in [200, 201]:
        print(f"Erro Plate Recognizer: {response.status_code} - {response.text}")
        return None

    resultado = response.json()

    if not resultado.get('results'):
        return None

    placa = resultado['results'][0]['plate'].upper()
    placa = re.sub(r'[^A-Z0-9]', '', placa)

    if len(placa) >= 7:
        return placa[:7]

    return None


def registrar_entrada(placa):
    try:
        response = requests.post(
            API_ENTRY,
            json={"placa": placa},
            timeout=5
        )

        if response.status_code == 200:
            print(f"[ENTRADA] {placa} registrada com sucesso.")
        else:
            print(f"[ENTRADA] {response.status_code} - {response.text}")

    except Exception as e:
        print(f"[ENTRADA] Erro: {e}")


def abrir_camera():
    tentativas = [
        (0, cv2.CAP_DSHOW),
        (1, cv2.CAP_DSHOW),
        (0, cv2.CAP_MSMF),
        (1, cv2.CAP_MSMF),
        (0, cv2.CAP_ANY),
        (1, cv2.CAP_ANY),
    ]

    for indice, backend in tentativas:
        cap = cv2.VideoCapture(indice, backend)

        if cap.isOpened():
            print(f"Câmera aberta no índice {indice}")
            return cap

        cap.release()

    return None


def confirmar_placa(recorte, placa):
    if recorte is not None and recorte.size > 0:
        recorte_exibicao = cv2.resize(recorte, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("Placa Capturada", recorte_exibicao)
        cv2.waitKey(1)
    else:
        print("Recorte inválido para exibição.")
        return

    if placa:
        print(f"\nPlaca lida pela IA: {placa}")
    else:
        print("\nA IA detectou a placa, mas não conseguiu ler o texto.")
        placa = ""

    confirmacao = input("A placa está correta? (S/N): ").upper().strip()

    if confirmacao == 'S':
        if placa_valida(placa):
            registrar_entrada(placa)
            print("Placa confirmada e registrada.")
        else:
            print("A placa lida pela IA está inválida.")
        return

    placa_corrigida = input("Digite a placa correta: ").upper().strip()
    placa_corrigida = re.sub(r'[^A-Z0-9]', '', placa_corrigida)

    if placa_valida(placa_corrigida):
        registrar_entrada(placa_corrigida)
        print(f"Placa corrigida e registrada: {placa_corrigida}")
    else:
        print("Placa inválida. Use o formato ABC1234 ou ABC1D23.")


# ================== MAIN ==================

def main():
    global clicou_capturar

    print("Leitor de Placas para Oficina")
    print("Clique em CAPTURAR para ler a placa.")
    print("Pressione Q para sair.")

    cap = abrir_camera()

    if cap is None:
        print("Não foi possível abrir nenhuma câmera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    nome_janela = "Leitor de Placas"
    cv2.namedWindow(nome_janela)
    cv2.setMouseCallback(nome_janela, clique_mouse)

    ultimo_recorte = None

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Erro ao capturar frame.")
            break

        frame_exibicao = frame.copy()

        ultimo_recorte = detectar_placa(frame_exibicao)

        desenhar_botao(frame_exibicao)

        cv2.putText(
            frame_exibicao,
            "A IA mostra a placa em verde. Clique em CAPTURAR para registrar.",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2
        )

        cv2.imshow(nome_janela, frame_exibicao)

        tecla = cv2.waitKey(1) & 0xFF

        if tecla == ord('q'):
            break

        if clicou_capturar:
            clicou_capturar = False

            if ultimo_recorte is None or ultimo_recorte.size == 0:
                print("Nenhuma placa válida detectada para capturar.")
                continue

            placa = extrair_texto_placa(ultimo_recorte)
            confirmar_placa(ultimo_recorte, placa)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
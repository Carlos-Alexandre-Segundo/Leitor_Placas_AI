from roboflow import Roboflow

# Configuração Inicial
API_KEY = "YOUR_OWN_API_KEY"      # <--- Insert your API key here Insira sua chave de API aqui
DATASET_URL = "URL_DATABASE_TRAIN"
FORMATO_EXPORTACAO = "yolov8"    # Choose any YOLO version, we'll use YoloV8.

# Código de Download
rf = Roboflow(api_key=API_KEY)
project = rf.workspace("YOUR_COUNTRY_PLATERECOGNIZER").project("YOUR_COUNTRY_PLATE_DETECTOR")
version = project.version(1)     # Número da versão do dataset (geralmente 1)
dataset = version.download(model_format=FORMATO_EXPORTACAO)

print("Download concluído! O dataset está em:", dataset.location)
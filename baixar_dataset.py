from roboflow import Roboflow

# Configuração Inicial
API_KEY = "8hL2HxTfnBzQfGIhLzB6"      # <--- Insira sua chave de API aqui
DATASET_URL = "https://universe.roboflow.com/brazilplatedetector/brazil-plates-detector"
FORMATO_EXPORTACAO = "yolov8"    # Escolha yolov8, yolov5, etc. Vamos usar o YOLOv8.

# Código de Download
rf = Roboflow(api_key=API_KEY)
project = rf.workspace("brazilplatedetector").project("brazil-plates-detector")
version = project.version(1)     # Número da versão do dataset (geralmente 1)
dataset = version.download(model_format=FORMATO_EXPORTACAO)

print("Download concluído! O dataset está em:", dataset.location)
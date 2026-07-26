import  cv2  #manejo de imagenes, deteccion de puntos clave, operaciones morfologicas y SIFT
import  numpy  as  np
import  os
import  glob
from  flask  import  Flask, request, jsonify  #crear servidor web
from  flask_cors  import  CORS  #peticiones api desde frontend
import  base64  #codificar y decondificar imagenes en formato texto
from  io  import  BytesIO  #manejar datos en binario como archivos
from  PIL  import  Image #abrir archivos binarios decodificados
import  json

app = Flask(__name__)
CORS(app)  

carpetaBd = "base_datos"
if not os.path.exists(carpetaBd):
    os.makedirs(carpetaBd)

def extraerDescriptoresSticker(rutaImg):
    #cargamos la imagen
    img = cv2.imread(rutaImg)
    if img is None:
        return None, None
    #2. filtro de blur para eliminar ruido
    imgBlur = cv2.GaussianBlur(img, (5, 5), 0)

    #prepara los pixeles para el algoritmo kmeans
    pixeles = np.float32(imgBlur.reshape((-1, 3)))
    criterio = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, etiquetas, _ = cv2.kmeans(pixeles, 2, None, criterio, 10, cv2.KMEANS_RANDOM_CENTERS)

    #rehacemos la imagen
    mapaEtiquetas = etiquetas.reshape((img.shape[0], img.shape[1]))

    #determinar cual etiqueta es el fondo
    bordes = np.concatenate([mapaEtiquetas[0, :], mapaEtiquetas[-1, :], 
                             mapaEtiquetas[:, 0], mapaEtiquetas[:, -1]])
    etiquetaFondo = np.bincount(bordes).argmax()

    #3. mascara kmeans
    mascaraKmeans = np.zeros_like(mapaEtiquetas, dtype=np.uint8)
    mascaraKmeans[mapaEtiquetas != etiquetaFondo] = 255

    #4. operaciones morfologicas
    kernel = np.ones((5, 5), np.uint8)
    mascaraKmeans = cv2.morphologyEx(mascaraKmeans, cv2.MORPH_OPEN, kernel)
    mascaraKmeans = cv2.morphologyEx(mascaraKmeans, cv2.MORPH_CLOSE, kernel)

    #5.Contornos
    contornos, _ = cv2.findContours(mascaraKmeans, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contornos:
        return None, None 
    mejorContorno = max(contornos, key=cv2.contourArea)

    #6.mascara solida
    mascaraSolida = np.zeros_like(mascaraKmeans)
    cv2.drawContours(mascaraSolida, [mejorContorno], -1, 255, cv2.FILLED)

    #7.recortar sticker y su mascara
    x, y, w, h = cv2.boundingRect(mejorContorno)
    stickerRecortado = img[y:y+h, x:x+w]
    mascaraRecortada = mascaraSolida[y:y+h, x:x+w]

    #imagen a escala de grises
    grisSticker = cv2.cvtColor(stickerRecortado, cv2.COLOR_BGR2GRAY)

    #8.obtencion de descriptores
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(grisSticker, mask=mascaraRecortada)
    return kp, des

# comunicacion con frontend
@app.route('/registrar', methods=['POST'])
def registrar():
    data = request.get_json()
    usuario = data.get('usuario')
    fotosB64 = data.get('fotos', [])

    if not usuario or len(fotosB64) != 5:
        return jsonify({'error': 'Se necesitan exactamente 5 fotos y un nombre de usuario'}), 400

    rutasTmp = []
    for i, imgB64 in enumerate(fotosB64):
        imgData = base64.b64decode(imgB64)
        img = Image.open(BytesIO(imgData))
        rutaTmp = f"tmp_{usuario}_{i}.jpg"
        img.save(rutaTmp)
        rutasTmp.append(rutaTmp)

    exito = logicaCrearHuella(rutasTmp, usuario)

    for r in rutasTmp:
        if os.path.exists(r):
            os.remove(r)

    if exito:
        return jsonify({'mensaje': f'Usuario {usuario} registrado correctamente'})
    else:
        return jsonify({'error': 'No se pudo procesar el sticker'}), 500

@app.route('/identificar', methods=['POST'])
def identificar():
    data = request.get_json()
    escenaB64 = data.get('escena')

    if not escenaB64:
        return jsonify({'error': 'Falta la imagen de la escena'}), 400

    imgData = base64.b64decode(escenaB64)
    img = Image.open(BytesIO(imgData))
    rutaTmp = "tmp_escena.jpg"
    img.save(rutaTmp)

    resultadoTexto, imgResultado = logicaIdentificarUsuario(rutaTmp)

    if os.path.exists(rutaTmp):
        os.remove(rutaTmp)

    imgB64 = None
    if imgResultado is not None:
        imgRgb = cv2.cvtColor(imgResultado, cv2.COLOR_BGR2RGB)
        pilImg = Image.fromarray(imgRgb)
        buffered = BytesIO()
        pilImg.save(buffered, format="JPEG")
        imgB64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    usuario = "Desconocido"
    puntos = 0
    if "si" in resultadoTexto:
        partes = resultadoTexto.split(":")
        if len(partes) > 1:
            usuarioPart = partes[1].split("(")
            usuario = usuarioPart[0].strip()
            if len(usuarioPart) > 1:
                puntos = usuarioPart[1].replace("puntos)", "").strip()
                try:
                    puntos = int(puntos)
                except:
                    puntos = 0

    return jsonify({
        'usuario': usuario,
        'puntos': puntos,
        'mensaje': resultadoTexto,
        'imagen': imgB64
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
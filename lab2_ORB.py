import os
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog

# Ruta del banco de imágenes
ruta_banco = r'C:\BD'

# Obtener la lista de carpetas
carpetas = ['FRESA', 'MANGO', 'MANZANA', 'PERA']

# Tamaño deseado para redimensionar las imágenes
nuevo_tamaño = (256, 256)

# Parámetros para HOG
orientaciones = 9
pixeles_por_celda = (8, 8)
celdas_por_bloque = (2, 2)

# Función para redimensionar una imagen y calcular sus características HOG y ORB
def preprocesar_imagen(ruta_imagen):
    # Leer la imagen
    imagen = cv2.imread(ruta_imagen)
    
    # Redimensionar la imagen
    imagen_redimensionada = cv2.resize(imagen, nuevo_tamaño)
    
    # Calcular el histograma de la imagen
    histograma = cv2.calcHist([imagen_redimensionada], [0], None, [256], [0, 256])
    
    # Calcular las características HOG de la imagen
    hog_features = hog(imagen_redimensionada, orientations=orientaciones, pixels_per_cell=pixeles_por_celda,
                       cells_per_block=celdas_por_bloque, channel_axis=-1)
    
    # Crear el detector ORB
    orb = cv2.ORB_create()
    
    # Encontrar los keypoints y descriptores ORB
    keypoints, descriptors = orb.detectAndCompute(imagen_redimensionada, None)
    
    return imagen_redimensionada, histograma, hog_features, keypoints, descriptors

# Procesar cada imagen en las carpetas
for carpeta in carpetas:
    # Ruta completa de la carpeta
    ruta_carpeta = os.path.join(ruta_banco, carpeta)
    
    # Obtener la lista de imágenes en la carpeta
    imagenes = os.listdir(ruta_carpeta)
    
    # Procesar cada imagen
    for imagen_nombre in imagenes:
        # Ruta completa de la imagen
        ruta_imagen = os.path.join(ruta_carpeta, imagen_nombre)
        
        # Preprocesar la imagen
        imagen_redimensionada, histograma, hog_features, keypoints, descriptors = preprocesar_imagen(ruta_imagen)
        
        # Mostrar la imagen redimensionada
        cv2.imshow('Imagen redimensionada', imagen_redimensionada)
        cv2.waitKey(0)
        
        # Mostrar el histograma
        plt.plot(histograma)
        plt.title('Histograma')
        plt.show()
        
        # Imprimir características HOG
        print('Características HOG:', hog_features)
        print()
        
        # Imprimir keypoints y descriptores ORB
        for i, keypoint in enumerate(keypoints):
            print('Keypoint', i+1, ':', keypoint)
            print('Descriptor', i+1, ':', descriptors[i])
            print()
        
# Cerrar las ventanas abiertas
cv2.destroyAllWindows()
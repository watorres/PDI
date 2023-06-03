import os
import cv2
import matplotlib.pyplot as plt

# Ruta del banco de imágenes
ruta_banco = r'C:\BD'

# Obtener la lista de carpetas
carpetas = ['FRESA', 'MANGO', 'MANZANA', 'PERA']

# Tamaño deseado para redimensionar las imágenes
nuevo_tamaño = (256, 256)

# Función para redimensionar una imagen y calcular su histograma
def preprocesar_imagen(ruta_imagen):
    # Leer la imagen
    imagen = cv2.imread(ruta_imagen)
    
    # Redimensionar la imagen
    imagen_redimensionada = cv2.resize(imagen, nuevo_tamaño)
    
    # Calcular el histograma de la imagen
    histograma = cv2.calcHist([imagen_redimensionada], [0], None, [256], [0, 256])
    
    return imagen_redimensionada, histograma

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
        imagen_redimensionada, histograma = preprocesar_imagen(ruta_imagen)
        
        # Mostrar la imagen redimensionada
        cv2.imshow('Imagen redimensionada', imagen_redimensionada)
        cv2.waitKey(0)
        
        # Mostrar el histograma
        plt.plot(histograma)
        plt.title('Histograma')
        plt.show()

# Cerrar las ventanas abiertas
cv2.destroyAllWindows()
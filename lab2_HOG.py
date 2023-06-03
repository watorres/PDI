import os
import cv2
import matplotlib.pyplot as plt
from skimage.feature import hog
from skimage import exposure

# Ruta del banco de imágenes
ruta_banco = r'C:\BD'

# Obtener la lista de carpetas
carpetas = ['FRESA', 'MANGO', 'MANZANA', 'PERA']

# Tamaño deseado para redimensionar las imágenes
nuevo_tamaño = (256, 256)

# Parámetros para HOG
orientaciones = 9
pixeles_por_celda = (8, 8)

# Función para redimensionar una imagen, calcular su histograma y extraer características HOG
def preprocesar_imagen(ruta_imagen):
    # Leer la imagen
    imagen = cv2.imread(ruta_imagen)
    
    # Redimensionar la imagen
    imagen_redimensionada = cv2.resize(imagen, nuevo_tamaño)
    
    # Calcular el histograma de la imagen
    histograma = cv2.calcHist([imagen_redimensionada], [0], None, [256], [0, 256])
    
    # Calcular las características HOG
    hog_features, hog_image = hog(imagen_redimensionada, orientations=orientaciones, pixels_per_cell=pixeles_por_celda,
                       visualize=True, channel_axis=-1)  # Utiliza `channel_axis` en lugar de `multichannel` en versiones futuras
    
    return imagen_redimensionada, histograma, hog_image

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
        imagen_redimensionada, histograma, hog_image = preprocesar_imagen(ruta_imagen)
        
        # Mostrar la imagen redimensionada
        plt.imshow(imagen_redimensionada)
        plt.title('Imagen redimensionada')
        plt.axis('off')
        plt.show()
        
        # Mostrar el histograma
        plt.plot(histograma)
        plt.title('Histograma')
        plt.show()
        
        # Mostrar la representación HOG
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.imshow(imagen_redimensionada)
        ax1.axis('off')
        ax1.set_title('Imagen redimensionada')

        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        ax2.imshow(hog_image_rescaled, cmap='gray')
        ax2.axis('off')
        ax2.set_title('Representación HOG')

        plt.show()
        
        # Pausa para ver la imagen antes de mostrar la siguiente
        input("Presiona Enter para continuar...")

# Cerrar las ventanas abiertas
cv2.destroyAllWindows()


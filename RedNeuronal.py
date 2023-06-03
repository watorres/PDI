import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, f1_score

# Ruta del banco de imágenes
ruta_banco = r'C:\BD'

# Obtener la lista de carpetas
carpetas = ['FRESA', 'MANGO', 'MANZANA', 'PERA']

# Tamaño deseado para redimensionar las imágenes
nuevo_tamaño = (256, 256)

# Parámetros para HOG
orientaciones = 9
pixeles_por_celda = (8, 8)

# Lista para almacenar las características HOG y las etiquetas
hog_features_list = []
etiquetas = []

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
        
        # Leer la imagen
        imagen = cv2.imread(ruta_imagen)
        
        # Redimensionar la imagen
        imagen_redimensionada = cv2.resize(imagen, nuevo_tamaño)
        
        # Calcular las características HOG
        hog_features = hog(imagen_redimensionada, orientations=orientaciones, pixels_per_cell=pixeles_por_celda,
                           channel_axis=-1)
        
        # Agregar las características HOG y la etiqueta a las listas
        hog_features_list.append(hog_features)
        etiquetas.append(carpeta)

# Convertir las listas a matrices numpy
X = np.array(hog_features_list)
y = np.array(etiquetas)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el clasificador de red neuronal
clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500)
clf.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = clf.predict(X_test)

# Calcular las métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
confusion = confusion_matrix(y_test, y_pred)
f_score = f1_score(y_test, y_pred, average='weighted')

# Imprimir los resultados
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Confusion Matrix:")
print(confusion)
print


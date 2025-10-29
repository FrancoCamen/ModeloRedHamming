import os
import shutil
import re

# Configura estas rutas según tus necesidades
DIRECTORIO_ORIGEN = "./"  # Cambia por tu ruta
DIRECTORIO_DESTINO = "./imagenes_filtradas"  # Cambia por tu ruta

def copiar_imagenes_filtradas():
    # Crear directorio destino si no existe
    if not os.path.exists(DIRECTORIO_DESTINO):
        os.makedirs(DIRECTORIO_DESTINO)
    
    # Patrón para archivos que terminan con _00000
    patron = re.compile(r'^\d{5}_00025_.*\.png$', re.IGNORECASE)
    
    contador = 0
    
    for archivo in os.listdir(DIRECTORIO_ORIGEN):
        if patron.match(archivo):
            origen = os.path.join(DIRECTORIO_ORIGEN, archivo)
            destino = os.path.join(DIRECTORIO_DESTINO, archivo)
            
            if os.path.isfile(origen):
                shutil.copy2(origen, destino)
                contador += 1
                print(f"Copiada: {archivo}")
    
    print(f"\nProceso completado. Se copiaron {contador} imágenes.")

if __name__ == "__main__":
    copiar_imagenes_filtradas()
# SymmetryAnnotationTool
Herramienta para añadir simetrías en objetos 3D, con el fin de generar un conjunto de objetos etiquetados correctamente para evaluar algoritmos de detección de simetrías.
## Instalación
1. Version de python 3.10.1
2. Crear un entorno virtual dentro de la carpeta del proyecto

```
cd SymmetryAnnotationTool
```
```
python -m venv venv
```
```
venv\Scripts\activate
```
3. Instalar dependencias
```
pip install -r requirements.txt
```
## Ejecución
```
py annotations_tool.py    
```
## Instrucciones de uso (Importante)
1. Al iniciar la herramienta aparecerá un objeto por defecto. Abrir un nuevo archivo .off de la carpeta offObjects.
   <br>
    ![import](https://github.com/ValeriaNahuelpan/SymmetryAnnotationTool/assets/62121145/db317928-5a45-4b32-b976-82346e3b455b)

2. Para agregar simetrías reflectivas se debe presionar "Add new" y colocar dos puntos con el click derecho sobre el objeto en lugares donde este se refleje. Se dibujará un plano perpendicular al vector normal que los une. Este plano se puede ocultar o 
    mostrar con el checkbox, guardar, eliminar o refinar.
   <br>
   ![addRef](https://github.com/ValeriaNahuelpan/SymmetryAnnotationTool/assets/62121145/51627fc5-c44d-491c-b8e3-b560b14d3e61)

3. Para agregar simetría rotacional (permite solo una), colocar más de 4 puntos (hasta 15) con el click derecho sobre el borde/órbita de algún objeto solido de revolución (Objetos que se crean al rotar una figura bidimensional alrededor de un eje formando un objeto tridimensional). Mientras más puntos se coloquen mejor es la 
   aproximación.
   Presionar "Draw rotation axis" y se dibujará un eje de rotación sobre el objeto. El eje se puede ocultar, guardar, eliminar o refinar.
   <br>
   ![addRot](https://github.com/ValeriaNahuelpan/SymmetryAnnotationTool/assets/62121145/75b028b2-236d-46bb-a742-00e987dfb4ef)
4. Al refinar simetrías estas aparecerán en el listado debajo de cada simetría no refinada. Refinar una simetría rotacional puede tomar varios minutos. 
   Al guardar una simetría refinada se eliminará la original y viceversa.
   <br>
   ![refining](https://github.com/ValeriaNahuelpan/SymmetryAnnotationTool/assets/62121145/d217eaa5-c5b5-41cf-bdc8-59f2aaa91cf7)
5. Los planos y ejes son representados por un punto y un vector normal, estos se almacenan en symmetries.json
## Encuesta sobre usabilidad
https://forms.gle/Q8m6nzwrqMRQ76rU9

## Contacto
telegram: @blckned

correo: valeria.nahuelpan@ug.uchile.cl

# SymmetryAnnotationTool
herramienta para etiquetar simetrías en objetos 3D
## Instalación
1. Version de python 3.10.1
2. Crear un entorno virtual dentro de la carpeta del proyecto

```
cd SymmetryAnnotationTool
```
```
python3 -m venv venv
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
## Uso
1. Al iniciar la herramienta aparecerá un objeto por defecto. Abrir un nuevo archivo .off de la carpeta offObjects.
   <br>
    ![import](https://github.com/ValeriaNahuelpan/SymmetryAnnotationTool/assets/62121145/db317928-5a45-4b32-b976-82346e3b455b)

2. Para agregar simetrías reflectivas se debe colocar dos puntos con el click derecho sobre el objeto luego de presionar "Add new" y se dibujará un plano perpendicular al vector normal que los une. Este plano se puede ocultar, guardar, eliminar o refinar.
![addRef](https://github.com/ValeriaNahuelpan/SymmetryAnnotationTool/assets/62121145/51627fc5-c44d-491c-b8e3-b560b14d3e61)

4. Para agregar simetría rotacional (permite solo una), colocar más de 4 puntos (hasta 15) con el click derecho sobre el borde/órbita de algún objeto circular. Mientras más puntos se coloquen mejor es la 
   aproximación.
   Presionar "Draw rotation axis" y se dibujará un eje de rotación sobre el objeto. El eje se puede ocultar, guardar, eliminar o refinar.
   <br>
   ![addRot](https://github.com/ValeriaNahuelpan/SymmetryAnnotationTool/assets/62121145/75b028b2-236d-46bb-a742-00e987dfb4ef)
5. Al refinar simetrías estas aparecerán en el listado debajo de cada simetría no refinada. Refinar una simetría rotacional puede tomar varios minutos. 
   Al guardar una simetría refinada se eliminará la original y viceversa.
   <br>
   ![refining](https://github.com/ValeriaNahuelpan/SymmetryAnnotationTool/assets/62121145/d217eaa5-c5b5-41cf-bdc8-59f2aaa91cf7)

## Encuesta sobre usabilidad
https://forms.gle/Q8m6nzwrqMRQ76rU9

## Contacto
telegram: @blckned

correo: valeria.nahuelpan@ug.uchile.cl

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
1. Abrir archivos .off de la carpeta offObjects
2. Para agregar simetrías reflectivas se debe colocar dos puntos con el click derecho sobre el objeto luego de presionar "Add new" y se dibujará un plano perpendicular al vector normal que los une.
   
![reflective](https://github.com/ValeriaNahuelpan/SymmetryAnnotationTool/assets/62121145/23b658bb-0b5f-4260-b728-efd0c432d07d)

3. Para agregar simetría rotacional (permite solo una), colocar más de 4 puntos (se puede hasta 15) con el click derecho sobre el borde de algún objeto circular. Presionar "Draw rotation axis" y se dibujará un eje de rotación sobre el objeto.

![rotationalAxis](https://github.com/ValeriaNahuelpan/SymmetryAnnotationTool/assets/62121145/8ca6b2ac-4172-4465-9c62-e3957ed2bc0f)

## Contacto
telegram: @blckned

correo: valeria.nahuelpan@ug.uchile.cl

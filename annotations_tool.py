# coding=utf-8
"""
Simple example using ImGui with GLFW and OpenGL
More info at:
https://pypi.org/project/imgui/
Installation:
pip install imgui[glfw]
Another example:
https://github.com/swistakm/pyimgui/blob/master/doc/examples/integrations_glfw3.py#L2
"""
import easygui
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import sys
import random
import imgui
from imgui.integrations.glfw import GlfwRenderer
import os.path
from reflective_symmetry import ReflectiveSymmetry
from rotational_symmetry import RotationalSymmetry
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grafica.gpu_shape import GPUShape
import grafica.transformations as tr
import grafica.basic_shapes as bs
import grafica.scene_graph as sg
import grafica.easy_shaders as es
import grafica.lighting_shaders as ls
import grafica.performance_monitor as pm
from grafica.assets_path import getAssetPath
import trimesh
from view_symmetry import *
import json
#import view_symmetry as vs
__author__ = "Valeria Nahuelpan"
__license__ = "MIT"

# A class to store the application control
class Controller:
    def __init__(self):
        self.fillPolygon = True
        self.showAxis = True

# we will use the global controller as communication with the callback function
controller = Controller()

def on_key(window, key, scancode, action, mods):

    if action != glfw.PRESS:
        return

    global controller

    if key == glfw.KEY_SPACE:
        controller.fillPolygon = not controller.fillPolygon

    elif key == glfw.KEY_ESCAPE:
        glfw.set_window_should_close(window, True)

    else:
        print('Unknown key')



global model
model = 'mesh13.off' # default

def readOFF(filename, color):
    vertices = []
    normals= []
    faces = []

    with open(filename, 'r') as file:
        line = file.readline().strip()
        assert line=="OFF"

        line = file.readline().strip()
        aux = line.split(' ')

        numVertices = int(aux[0])
        numFaces = int(aux[1])

        for i in range(numVertices):
            aux = file.readline().strip().split(' ')
            vertices += [float(coord) for coord in aux[0:]]

        vertices = np.asarray(vertices)
        vertices = np.reshape(vertices, (numVertices, 3))
        print(f'Vertices shape: {vertices.shape}')

        normals = np.zeros((numVertices,3), dtype=np.float32)
        print(f'Normals shape: {normals.shape}')

        for i in range(numFaces):
            aux = file.readline().strip().split(' ')
            aux = [int(index) for index in aux[0:]]
            faces += [aux[1:]]

            vecA = [vertices[aux[2]][0] - vertices[aux[1]][0], vertices[aux[2]][1] - vertices[aux[1]][1], vertices[aux[2]][2] - vertices[aux[1]][2]]
            vecB = [vertices[aux[3]][0] - vertices[aux[2]][0], vertices[aux[3]][1] - vertices[aux[2]][1], vertices[aux[3]][2] - vertices[aux[2]][2]]

            res = np.cross(vecA, vecB)
            normals[aux[1]][0] += res[0]
            normals[aux[1]][1] += res[1]
            normals[aux[1]][2] += res[2]

            normals[aux[2]][0] += res[0]
            normals[aux[2]][1] += res[1]
            normals[aux[2]][2] += res[2]

            normals[aux[3]][0] += res[0]
            normals[aux[3]][1] += res[1]
            normals[aux[3]][2] += res[2]

        norms = np.linalg.norm(normals,axis=1)
        normals = normals/norms[:,None]

        color = np.asarray(color)
        color = np.tile(color, (numVertices, 1))

        vertexData = np.concatenate((vertices, color), axis=1)
        vertexData = np.concatenate((vertexData, normals), axis=1)

        indices = []
        vertexDataF = []
        index = 0

        for face in faces:
            vertex = vertexData[face[0],:]
            vertexDataF += vertex.tolist()
            vertex = vertexData[face[1],:]
            vertexDataF += vertex.tolist()
            vertex = vertexData[face[2],:]
            vertexDataF += vertex.tolist()

            indices += [index, index + 1, index + 2]
            index += 3

        return bs.Shape(vertexDataF, indices), vertices, faces

def createOFFShape(pipeline, model, r,g, b):
    shape, vertices, faces = readOFF(getAssetPath(model), (r, g, b))
    gpuShape = es.GPUShape().initBuffers()
    pipeline.setupVAO(gpuShape)
    gpuShape.fillBuffers(shape.vertices, shape.indices, GL_STATIC_DRAW)

    return gpuShape, vertices, faces

color = [1.2,1.2,1.2]

def createScene(pipeline):
    global obj_vertices, obj_faces, mesh, sphereNode, transformed_mesh

    ObjShape, obj_vertices, obj_faces = createOFFShape(pipeline, model, color[0],color[1],color[2])

    # we must adjust the mesh to match the object
    obj_faces = np.flip(obj_faces, axis=1)
    mesh = trimesh.Trimesh(vertices=obj_vertices, faces=obj_faces)
    rotation_matrix = trimesh.transformations.rotation_matrix(np.pi, [1, 0, 0])
    mesh.apply_transform(rotation_matrix)
    reflect_matrix = np.array([
        [-1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    mesh.apply_transform(reflect_matrix)
    angle = np.pi
    rotation_matrix = np.array([[np.cos(angle), 0, np.sin(angle), 0],
                                [0, 1, 0, 0],
                                [-np.sin(angle), 0, np.cos(angle), 0],
                                [0, 0, 0, 1]])

    mesh.apply_transform(rotation_matrix)
    transformed_mesh = mesh.copy()

    ObjNode = sg.SceneGraphNode("ObjNode")
    ObjNode.childs += [ObjShape]
    Obj2Node = sg.SceneGraphNode("ObjNode")
    Obj2Node.childs += [ObjNode]
    ObjRotation = sg.SceneGraphNode("ObjRotation")
    ObjRotation.childs += [Obj2Node]
    SceneNode= sg.SceneGraphNode("SceneNode")
    SceneNode.childs += [ObjRotation]

    return SceneNode

def createPlane(pipeline):
    SceneNode= sg.SceneGraphNode("SceneNode")
    return SceneNode

inv_rotation_matrix = np.identity(4)
mesh_transform = np.identity(4)

# Función de devolución de llamada que se llama cuando se presiona un botón del mouse
def mouse_button_callback(window, button, action, mods):
    global isDragging, colors
    global transformed_mesh, inv_rotation_matrix, hit_location
    if button == glfw.MOUSE_BUTTON_LEFT:
        if action == glfw.PRESS:
            isDragging = True
        elif action == glfw.RELEASE:
            isDragging = False

    if button == glfw.MOUSE_BUTTON_RIGHT and action ==glfw.PRESS :
        direction = ray_direction
        transformed_mesh.apply_transform(mesh_transform)

        # Lanza el rayo desde una posición fija en el espacio de tu objeto en la dirección calculada
        ray_origin = [0, 0,  0.5]
        hit_location, _, _ = transformed_mesh.ray.intersects_location(
            ray_origins=[ray_origin],
            ray_directions=[direction],
            multiple_hits=False
        )

        if len(hit_location)>0 and refSymmetry.adding:
            refSymmetry.create_point(pipeline, Scene, (hit_location[0][0], -hit_location[0][1], hit_location[0][2]), inv_rotation_matrix)
            lastPair = refSymmetry.pointsPairs[-1]
            # if pair of points is completed
            if len(lastPair) == 2:
                if len(refSymmetry.info)>= len(refSymmetry.colors):
                    refSymmetry.colors.extend(refSymmetry.colors)
                cubeShape, _, _ = createOFFShape(pipeline, 'cube.off', refSymmetry.colors[len(refSymmetry.info)][0], refSymmetry.colors[len(refSymmetry.info)][1], refSymmetry.colors[len(refSymmetry.info)][2])
                # create the plane of last tuple added
                planeScene = sg.findNode(Plane, "SceneNode")
                plane = sg.SceneGraphNode("planeRef" + str(1+len(refSymmetry.info)))
                plane.transform = tr.uniformScale(0.00)
                plane.childs += [cubeShape]
                planeScene.childs += [plane]
                # points positions
                point_p = lastPair[0].transform[:,3]
                point_q = lastPair[1].transform[:,3]
                # midpoint and normalized vector calculation
                punto_medio = [(point_p[0] + point_q[0]) / 2, (point_p[1] + point_q[1]) / 2, (point_p[2] + point_q[2]) / 2]
                normal = ((point_p[0] - point_q[0]), (point_p[1] - point_q[1]), (point_p[2] - point_q[2]))
                magnitud = (normal[0]**2 + normal[1]**2 + normal[2]**2)**0.5
                normalizado = [normal[0] / magnitud, normal[1] / magnitud, normal[2] / magnitud]
                # creation of the plane and its transformation matrix
                symmetryPlane = SymmetryPlane(np.array(punto_medio), np.array(normalizado))
                # plane = sg.findNode(Plane, "planeNode")
                plane.transform = tr.matmul([inv_rotation_matrix,tr.rotationX(locationRX),tr.rotationY(locationRY),symmetryPlane.transform, tr.scale(0,0.2,0.5)])
                refSymmetry.save_symmetry(punto_medio, normalizado, [])
        elif len(hit_location)>0 and rotSymmetry.adding:
            rotSymmetry.create_point(pipeline, Scene, (hit_location[0][0], -hit_location[0][1], hit_location[0][2]), inv_rotation_matrix)
        else:
            print("out of object")
        transformed_mesh = mesh.copy()

# movimiento del objeto con el cursor
lastRY = 0.0
xDirection = 0  # -1 para movimiento hacia izquierda, 1 para movimiento hacia derecha
lastRX = 0.0
yDirection = 0  # -1 para movimiento hacia abajo, 1 para movimiento hacia arriba
def cursor_position_callback(window, xpos, ypos):
    global  initialX, initialY, xDirection, yDirection
    global isDragging, locationRY, locationRX, inv_rotation_matrix, mesh_transform
    if isDragging:
        i_posX = initialX
        i_posY = initialY
        # actualiza direccion del movimiento en el eje X y Y
        if xpos > initialX:
            xDirection = 1
        elif xpos < initialX:
            xDirection = -1
        else:
            xDirection = 0

        if ypos > initialY:
            yDirection = -1
        elif ypos < initialY:
            yDirection = 1
        else:
            yDirection = 0

        # actualiza locationRX y locationRY segun la direccion del movimiento en el eje X y Y

        if xDirection == 0:
            locationRY = locationRY
        elif xDirection == 1:
            locationRY = locationRY + (xpos - i_posX) * 0.0005
        else:
            locationRY = locationRY - (i_posX - xpos) * 0.0005

        if yDirection == 0:
            locationRX = locationRX
        elif yDirection == 1:
            locationRX = locationRX + (ypos- i_posY) * 0.0005
        else:
            locationRX = locationRX - (i_posY -  ypos) * 0.0005

        #aplicamos la transformacion segun movimiento del cursor
        objRot.transform = tr.matmul([tr.rotationX(locationRX),tr.rotationY(locationRY)])
        # movemos la escena que tiene los planos junto al objeto
        planeNodeScene = sg.findNode(Plane,"SceneNode")
        planeNodeScene.transform = tr.matmul([tr.rotationX(locationRX),tr.rotationY(locationRY)])
        # inversa de la matriz de transformacion del objeto pq rotaba junto a sus ejes entonces se necesita devolver los ejes
        inv_rotation_matrix = np.linalg.inv(objRot.transform)
        mesh_transform = tr.matmul([tr.rotationX(-locationRX),tr.rotationY(locationRY)])


    else:
        initialX = xpos
        initialY = ypos


def object_exists(json_data, file_name):
    data = json.loads(json_data)
    objects = data["objects"]
    for obj in objects:
        filename = fr"{obj['file']}"
        if filename == file_name.replace("\\","/"):
            return True
    return False


def getSymmetriesJson():
        proyectPath = os.path.dirname(os.path.abspath("annotations_tool.py"))
        relativPath = os.path.relpath(modelobj, proyectPath)
        # read JSON
        with open("symmetries.json", "r") as json_file:
            json_data = json_file.read()

        exists = object_exists(json_data, relativPath)

        if exists:
            data = json.loads(json_data)
            objects = data["objects"]
            for obj in objects:
                if obj["file"] == relativPath.replace("\\","/"):
                    symmetries = obj["symmetries"] # reflectives and rotational
            return symmetries

        else:
            return {"rotational":{}, "reflectives": []}

checkbox_states = {"reflectives":[],"rotational":[], "refined_reflectives":[],"refined_rotational":[]}
def symmetriesGUI(symmetries, type):
        global symmetriesJson, Plane, refSymmetry
        proyectPath = os.path.dirname(os.path.abspath("annotations_tool.py"))
        relativPath = os.path.relpath(modelobj, proyectPath).replace("\\","/") #folder-dataset/object.obj
        to_delete = []
        # this function shows the symmetries of each object
        for i in range(len(symmetries)):
            if type == "reflectives":
                if symmetries[i]["isRef"]:
                    imgui.text_colored("refined symmetry" + str(i+1), refSymmetry.colors[i][0],refSymmetry.colors[i][1],refSymmetry.colors[i][2])
                else:
                    imgui.text_colored("symmetry" + str(i+1), refSymmetry.colors[i][0],refSymmetry.colors[i][1],refSymmetry.colors[i][2])

            else:
                if symmetries[i]["isRef"]:
                    imgui.text("refined symmetry"+ str(i+1))
                else:
                    imgui.text("symmetry"+ str(i+1))
            imgui.same_line()

            # DISPLAY
            try: # we use try because the update of the variable "i" is not immediate
                #checkbox initialized in true
                if i >= len(checkbox_states[type]):
                    checkbox_states[type].append(True)
                # Checkbox to display symmetry
                _, checkbox_states[type][i] = imgui.checkbox(f"##checkbox_{i}", checkbox_states[type][i])
                if type == "reflectives":
                    if symmetries[i]["isRef"]:
                        p = sg.findNode(Plane,"planeRefined" + str(i+1))
                    else:
                        p = sg.findNode(Plane,"planeRef" + str(i+1))
                    if checkbox_states[type][i] == False:
                        p.transform = tr.uniformScale(0.00)
                    else:
                        vsym = SymmetryPlane(np.array(symmetries[i]["point"]),symmetries[i]["normal"])
                        p.transform = tr.matmul([inv_rotation_matrix,tr.rotationX(locationRX),tr.rotationY(locationRY),vsym.transform, tr.scale(0,0.2,0.4)])
                else:
                    if symmetries[i]["isRef"]:
                        axis = sg.findNode(Scene,"axisRefined")
                    else:
                        axis = sg.findNode(Scene,"axis")
                    #axis = sg.findNode(Scene,"axis")
                    if checkbox_states[type][i] == False:
                        axis.transform = tr.uniformScale(0.00)
                    else:
                        vsym =  SymmetryPlane(np.array(symmetries[i]["point"]), symmetries[i]["normal"])
                        axis.transform = tr.matmul([vsym.transform, tr.scale(5,0.001,0.001)])
                if imgui.is_item_hovered():
                    imgui.begin_tooltip()
                    imgui.text("Display/Hide")
                    imgui.end_tooltip()
            except:
                pass

            # DELETE
            imgui.same_line()
            imgui.push_style_color(imgui.COLOR_BUTTON, 0.3, 0.0, 0)
            imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.5, 0.0, 0)
            imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.5, 0.0, 0)
            if imgui.button(f"X##symmetry_{i}", 20):
                if type == "reflectives":
                    # change de name of plane and removed plane from scene
                    if symmetries[i]["isRef"]:
                        planeactualRef= sg.findNode(Plane,"planeRefined" + str(i+1))
                        planeactualRef.name = "removed"
                        planeactualRef.transform = tr.uniformScale(0.00)
                    else:
                        planeactual= sg.findNode(Plane,"planeRef" + str(i+1))
                        planeactual.name = "removed"
                        planeactual.transform = tr.uniformScale(0.00)

                    # the names of the remaining planes are changed only if symetry has not been refined.
                    for index in range(i+2, len(symmetries)+1):
                        if symmetries[i]["isRef"] or len(symmetries[i]["normalRefined"])==0:
                            try:
                                pRef= sg.findNode(Plane, "planeRefined" + str(index))
                                pRef.name = "planeRefined" + str(index-1)
                            except:
                                pass
                            try:
                                plane= sg.findNode(Plane, "planeRef" + str(index))
                                plane.name = "planeRef" + str(index-1)
                            except:
                                pass
                else: #rotational
                    if symmetries[i]["isRef"]:
                        axis = sg.findNode(Scene, "axisRefined")
                        axis.name ="removedAxisRefined"
                        axis.transform = tr.uniformScale(0.00)
                        rotSymmetry.axisAdded = False
                        rotSymmetry.points = np.array([])
                    else:
                        axis = sg.findNode(Scene, "axis")
                        axis.name ="removedAxis"
                        axis.transform = tr.uniformScale(0.00)
                        rotSymmetry.axisAdded = False
                        rotSymmetry.points = np.array([])
                # remove symmetry from json if its saved
                with open('symmetries.json', 'r') as f:
                    content_json = json.load(f)
                for objeto in content_json['objects']:
                    if objeto['file'] == relativPath:
                        symmetriesList = objeto['symmetries'][type]
                        symmetriesList[:] = [symmetry for symmetry in symmetriesList if symmetry != {"point":symmetries[i]["point"],"normal":symmetries[i]["normal"]}]
                        break
                updated_content = json.dumps(content_json, indent=4)
                with open('symmetries.json', 'w') as f:
                    f.write(updated_content)
                # update json info
                symmetriesJson = getSymmetriesJson()
                # readjust the list of colors and the color of the labels
                if type == "reflectives" and (len(symmetries[i]["normalRefined"])==0 or symmetries[i]["isRef"]):
                    color = refSymmetry.colors.pop(i)
                    refSymmetry.colors.append(color)
                # remove symmetry from Class ReflectiveSymmetry or RotationalSymmetry
                #symmetries.pop(i)
                if len(symmetries[i]["normalRefined"])==0 or symmetries[i]["isRef"]:
                    to_delete.append(i)
                else:
                    symmetries[i]["point"] = symmetries[i]["pointRefined"]
                    symmetries[i]["normal"] = symmetries[i]["normalRefined"]
                    #symmetries[i]["normalRefined"] = [0]
                    symmetries[i]["isRef"] = True

            if imgui.is_item_hovered():
                imgui.begin_tooltip()
                imgui.text("Remove")
                imgui.end_tooltip()
            imgui.pop_style_color(3)

            # SAVE
            # if it's not saved on the json
            if json.dumps({"point":symmetries[i]["point"],"normal":symmetries[i]["normal"]}) not in json.dumps(symmetriesJson[type]):
                imgui.same_line()
                if imgui.button(f"Save##symmetry_{i}"):
                    with open('symmetries.json', 'r') as f:
                        content_json = json.load(f)
                    object_exist = None
                    for obj in content_json['objects']:
                        if obj['file'] == relativPath:
                            object_exist = obj
                            break
                    # if object exists
                    if object_exist is not None:
                        new_reflective = symmetries[i]
                        object_exist['symmetries'][type].append({"point": new_reflective["point"], "normal": new_reflective["normal"]})
                    else:
                        new_object = {
                            "file": relativPath,
                            "symmetries": {
                                "rotational": [],
                                "reflectives": []
                            }
                        }
                        new_object['symmetries'][type].append({"point":symmetries[i]["point"],"normal":symmetries[i]["normal"]})
                        content_json['objects'].append(new_object)
                    updated_content = json.dumps(content_json, indent=4)
                    with open('symmetries.json', 'w') as archivo:
                        archivo.write(updated_content)
                    print("The new reflective symmetry has been successfully added in the JSON file.")
                    if len(symmetries[i]["normalRefined"]) > 1 and symmetries[i]["isRef"]==False:
                        symmetries[i]["normalRefined"] = []
                        symmetries[i]["pointRefined"] = []
                        symmetries[i]["isRef"] = False
                        if type=="reflectives":
                            pRef= sg.findNode(Plane,"planeRefined" + str(i+1))
                        else:
                            pRef= sg.findNode(Scene,"axisRefined")
                        pRef.name = "removed"
                        pRef.transform = tr.uniformScale(0.00)
                    symmetriesJson = getSymmetriesJson()

            # REFINE
            if len(symmetries[i]["normalRefined"]) == 0:
                imgui.same_line()
                if imgui.button(f"Refine##symmetry_{i}"):
                    #Read triangle mesh with open3d
                    mesh = o3d.io.read_triangle_mesh(relativPath)
                    #Convert mesh to numpy arrays
                    points = np.asarray(mesh.vertices)
                    triangles = np.asarray(mesh.triangles)
                    #Refinamos la simetrias reflectiva
                    if type=="reflectives":
                        new_normal = refineReflectionTransform(symmetries[i]["point"], symmetries[i]["normal"], points)
                        #create a new plane with the refined symmetry
                        sym_ref = SymmetryPlane(np.array(symmetries[i]["point"]), np.array(new_normal))
                        cubeShape, _, _ = createOFFShape(pipeline, 'cube.off', 1,1,1)
                        planeScene = sg.findNode(Plane, "SceneNode")
                        pRef = sg.SceneGraphNode("planeRefined" + str(i+1))
                        pRef.childs += [cubeShape]
                        pRef.transform = tr.matmul([inv_rotation_matrix,tr.rotationX(locationRX),tr.rotationY(locationRY), sym_ref.transform, tr.scale(0,0.2,0.4)])
                        planeScene.childs += [pRef]
                        symmetries[i]["pointRefined"] = symmetries[i]["point"]
                    else: # rotational
                        new_point, new_normal = refineRotationTransform(np.array(symmetries[i]["point"]), np.array(symmetries[i]["normal"]), points)
                        sym_rot = SymmetryPlane(point=new_point, normal=new_normal)
                        cubeShape, _, _ = createOFFShape(pipeline, 'cube.off', 1,1,1)
                        rotationAxis = sg.SceneGraphNode("axisRefined")
                        rotationAxis.childs += [cubeShape]
                        rotationAxis.transform = tr.matmul([sym_rot.transform, tr.scale(5,0.001,0.001)])
                        Scene.childs += [rotationAxis]
                        symmetries[i]["pointRefined"] = [new_point[0],new_point[1],new_point[2]]
                    # save new normal in the class
                    symmetries[i]["normalRefined"] = [new_normal[0],new_normal[1],new_normal[2]]

            # symmetry has been refined
            elif symmetries[i]['isRef']==False:
                if type == "reflectives":
                    imgui.text_colored("refined symmetry" + str(i+1) , refSymmetry.colors[i][0],refSymmetry.colors[i][1],refSymmetry.colors[i][2])
                else:
                    imgui.text("refined symmetry" + str(i+1))
                imgui.same_line()
                # DISPLAY
                try:
                    #checkbox initialized in true
                    if i >= len(checkbox_states["refined_" + str(type)]):
                        checkbox_states["refined_" + str(type)].append(True)
                    # Checkbox to display symmetry
                    _, checkbox_states["refined_" + str(type)][i] = imgui.checkbox(f"##checkboxRefined_{str(i) + str(type)}", checkbox_states["refined_" + str(type)][i])
                    if type == "reflectives":
                        pRef = sg.findNode(Plane,"planeRefined" + str(i+1))
                        if checkbox_states["refined_" + str(type)][i] == False:
                            pRef.transform = tr.uniformScale(0.00)
                        else:
                            sym_ref = SymmetryPlane(np.array(symmetries[i]["point"]), np.array(symmetries[i]["normalRefined"]))
                            pRef.transform = tr.matmul([inv_rotation_matrix,tr.rotationX(locationRX),tr.rotationY(locationRY), sym_ref.transform, tr.scale(0,0.2,0.4)])

                    else:
                        axis = sg.findNode(Scene,"axisRefined")
                        if checkbox_states["refined_" + str(type)][i] == False:
                            axis.transform = tr.uniformScale(0.00)
                        else:
                            vsym =  SymmetryPlane(np.array(symmetries[i]["pointRefined"]), symmetries[i]["normalRefined"])
                            axis.transform = tr.matmul([vsym.transform, tr.scale(5,0.001,0.001)])
                    if imgui.is_item_hovered():
                        imgui.begin_tooltip()
                        imgui.text("Display/Hide")
                        imgui.end_tooltip()
                except:
                    pass
                imgui.same_line()
                imgui.push_style_color(imgui.COLOR_BUTTON, 0.3, 0.0, 0)
                imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, 0.5, 0.0, 0)
                imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, 0.5, 0.0, 0)
                if imgui.button(f"X##XRefine_{i}"):
                    # remove the refined symmetry inside the class
                    symmetries[i]["pointRefined"] = []
                    symmetries[i]["normalRefined"] = []
                    symmetries[i]["isRef"] = False
                    if type=="reflectives":
                        pRef= sg.findNode(Plane,"planeRefined" + str(i+1))
                    else:
                        pRef= sg.findNode(Scene,"axisRefined")
                    pRef.name = "removed"
                    pRef.transform = tr.uniformScale(0.00)
                if imgui.is_item_hovered():
                    imgui.begin_tooltip()
                    imgui.text("Remove")
                    imgui.end_tooltip()
                imgui.pop_style_color(3)
                imgui.same_line()
                if imgui.button(f"Save##saveRefine{i}"):
                    # remove unrefined symmetry from scene and json
                    if type=="reflectives":
                        sym= sg.findNode(Plane,"planeRef" + str(i+1))
                    else:
                        sym= sg.findNode(Scene,"axis")
                    sym.name = "removed"
                    sym.transform = tr.uniformScale(0.00)
                    # remove symmetry from json if its saved
                    with open('symmetries.json', 'r') as f:
                        content_json = json.load(f)
                    for objeto in content_json['objects']:
                        if objeto['file'] == relativPath:
                            symmetriesList = objeto['symmetries'][type]
                            symmetriesList[:] = [symmetry for symmetry in symmetriesList if symmetry != {"point":symmetries[i]["point"],"normal":symmetries[i]["normal"]}]
                            break
                    updated_content = json.dumps(content_json, indent=4)
                    with open('symmetries.json', 'w') as f:
                        f.write(updated_content)
                    symmetries[i]["point"] = symmetries[i]["pointRefined"]
                    symmetries[i]["normal"] = symmetries[i]["normalRefined"]
                    #symmetries[i]["normalRefined"] = [0]
                    symmetries[i]["isRef"] = True
                    # update json info
                    symmetriesJson = getSymmetriesJson()
                    # save the refined in the JSON #!
                    for obj in content_json['objects']:
                        if obj['file'] == relativPath:
                            object_exist = obj
                            break
                    # if object exists
                    if object_exist is not None:
                        new_reflective = symmetries[i]
                        object_exist['symmetries'][type].append({"point": new_reflective["point"], "normal": new_reflective["normal"]})
                    else:
                        new_object = {
                            "file": relativPath,
                            "symmetries": {
                                "rotational": [],
                                "reflectives": []
                            }
                        }
                        new_object['symmetries'][type].append({"point":symmetries[i]["point"],"normal":symmetries[i]["normal"]})
                        content_json['objects'].append(new_object)
                    updated_content = json.dumps(content_json, indent=4)
                    with open('symmetries.json', 'w') as archivo:
                        archivo.write(updated_content)
                    print("The new reflective symmetry has been successfully added in the JSON file.")
                    # update json info
                    symmetriesJson = getSymmetriesJson()


        for i in reversed(to_delete):
            symmetries.pop(i)



rotSymmetry = RotationalSymmetry(15)
refSymmetry = ReflectiveSymmetry()
open_file = False
colors = [
    (1.0, 0.0, 1.0),   # Fucsia
    (0.0, 0.75, 1.0),  # Celeste
    (1.0, 0.0, 0.0),   # Rojo
    (1.0, 1.0, 0.0),   # Amarillo
    (0.0, 1.0, 0.0),   # Verde
    (0.0, 0.0, 1.0),   # Azul
    (1.0, 0.5, 0.0),   # Naranja
    (0.5, 0.0, 1.0),   # Morado
    (1.0, 0.75, 0.8),  # Rosa
    (0.0, 1.0, 0.75)   # Turquesa
]
modelobj = "complete-dataset/0902.obj" #by default
def transformGuiOverlay(locationRX, locationRY, locationRZ):
    global model, modelobj, relativPath, symmetriesJson
    global objRot, Scene, Plane, pipeline, open_file
    global controller, rotSymmetry, refSymmetry

    # start new frame context
    imgui.new_frame()

    # open new window context
    imgui.set_next_window_position(10, 20)
    imgui.begin("Labeled symmetries", False, imgui.WINDOW_ALWAYS_AUTO_RESIZE)

    if imgui.begin_main_menu_bar():
        if imgui.begin_menu("File", True):
            clicked_quit, _ = imgui.menu_item("import new object (.off)", 'Ctrl+I', False, True)
            if clicked_quit:
                modelobj = easygui.fileopenbox() #.off
                rotSymmetry = RotationalSymmetry(15)
                refSymmetry = ReflectiveSymmetry()
                # obj2off(modelobj, getAssetPath('newOff.off') )   #convert .obj to .off
                model = getAssetPath(modelobj) #.off
                # new scene
                Scene = createScene(pipeline)
                # new plane scene
                Plane = createPlane(pipeline2)
                # if object have symmetry it is shown
                symmetriesJson = getSymmetriesJson()
                proyectPath = os.path.dirname(os.path.abspath("annotations_tool.py"))
                relativPath = os.path.relpath(modelobj, proyectPath)
                # drawing symmetries reflectives and rotationals from json
                symmetry_count = 1
                for symmetry in symmetriesJson["reflectives"]:
                    vsym =  SymmetryPlane(np.array(symmetry["point"]), symmetry["normal"])
                    cubeShape, _, _ = createOFFShape(pipeline, 'cube.off', colors[symmetry_count-1][0],colors[symmetry_count-1][1], colors[symmetry_count-1][2])
                    # create the plane of last tuple added
                    planeScene = sg.findNode(Plane, "SceneNode")
                    plane = sg.SceneGraphNode("planeRef" + str(symmetry_count))
                    plane.transform = tr.uniformScale(0.00)
                    plane.childs += [cubeShape]
                    planeScene.childs += [plane]
                    # plane = sg.findNode(Plane, "planeNode")
                    plane.transform = tr.matmul([inv_rotation_matrix,tr.rotationX(locationRX),tr.rotationY(locationRY),vsym.transform, tr.scale(0,0.2,0.4)])
                    refSymmetry.save_symmetry(symmetry["point"], symmetry["normal"])
                    symmetry_count += 1

                for symmetry in symmetriesJson["rotational"]:
                    rotSymmetry.save_symmetry(symmetry["point"],symmetry["normal"])
                    rotSymmetry.drawAxis(pipeline, Scene)
                open_file = True
            imgui.end_menu()
        imgui.end_main_menu_bar()
    if open_file:
        # list of symmetries
        if imgui.tree_node("Reflectives           ", imgui.TREE_NODE_SELECTED): # flags: https://pyimgui.readthedocs.io/en/latest/guide/treenode-flags.html#treenode-flag-options
            symmetriesGUI(refSymmetry.info,"reflectives")
            if refSymmetry.adding:
                imgui.text("Adding Symmetry ...")
            else:
                if not rotSymmetry.adding:
                    if imgui.button("Add new##reflective"):
                        refSymmetry.adding = True
            imgui.tree_pop()
        if imgui.tree_node("Rotational           ", imgui.TREE_NODE_SELECTED):
            symmetriesGUI(rotSymmetry.info, "rotational")
            if rotSymmetry.adding == False and rotSymmetry.axisAdded == False and len(rotSymmetry.info)<1 and not refSymmetry.adding:
                if imgui.button("Add new##rotational"):
                    rotSymmetry.adding = True
            if rotSymmetry.adding == True and rotSymmetry.axisAdded == False and len(rotSymmetry.points)>4:
                if imgui.button("Draw rotation axis"):
                    Scene.childs = Scene.childs[:-(len(rotSymmetry.points))]
                    rotSymmetry.calculate_plane(pipeline, Scene)
                    rotSymmetry.adding = False
                    rotSymmetry.axisAdded = True
            if rotSymmetry.adding==True:
                imgui.text("Adding Symmetry ...")
            imgui.tree_pop()
    else:
        imgui.text("Open a new file from the menu!!")

    imgui.end()
    # pass all drawing comands to the rendering pipeline
    # and close frame context
    imgui.render()
    imgui.end_frame()

    return locationRX, locationRY, locationRZ


# Function to convert the cursor position in the GLFW window to a ray direction in object space
def get_ray_direction(window, x, y):
    # Cursor position in pixels
    x, y = glfw.get_cursor_pos(window)
    # Gets the width and height of the window
    width, height = glfw.get_framebuffer_size(window)
    # Cursor coordinates are normalized
    try:
        cursor_pos = np.array([x/width, y/height])
          #ZeroDivisionError: float division by zero
        # The inverse projection matrix and the inverse view matrix are obtained
        inv_projection = np.linalg.inv(projection)
        inv_view = np.linalg.inv(view)
        # Converts cursor position from normalized coordinates to view space coordinates
        view_pos = inv_projection @ np.array([cursor_pos[0]*2-1, cursor_pos[1]*2-1, -1, 1])
        view_pos /= view_pos[3]
        # Converts cursor position from view space to world space
        world_pos = inv_view @ view_pos
        # Calculate the direction of the ray
        ray_dir = world_pos[:3] - viewPos
        ray_dir /= np.linalg.norm(ray_dir)
        ray_direction = ray_dir
        return ray_direction
    except:
        pass

direction = []
if __name__ == "__main__":

    # Initialize glfw
    if not glfw.init():
        glfw.set_window_should_close(window, True)
    glfw.window_hint(glfw.RESIZABLE, glfw.FALSE)
    glfw.window_hint(glfw.MAXIMIZED, glfw.FALSE)
    width = 800
    height = 800
    title = "Symmetry Annotations tool"
    window = glfw.create_window(width, height,title , None, None)
    glfw.make_context_current(window)
    windowWidth, windowHeight = glfw.get_window_size(window)
    glfw.window_hint(glfw.SCALE_TO_MONITOR, True)
    glViewport(0, 0, windowWidth, windowHeight)


    # Global variables for object movement
    lastX, lastY = 0, 0
    isDragging = False
    initialX = 0
    initialY = 0
    symmetriesJson = getSymmetriesJson()
    if not window:
        glfw.terminate()
        glfw.set_window_should_close(window, True)

    glfw.make_context_current(window)
    # Connecting the callback function 'on_key' to handle keyboard events
    glfw.set_key_callback(window, on_key)

    # Assembling the shader program (pipeline) with both shaders
    mvpPipeline = es.SimpleModelViewProjectionShaderProgram()
    pipeline = ls.SimpleFlatShaderProgram()
    pipeline2 = ls.SimpleFlatBlendShaderProgram()

    # Telling OpenGL to use our shader program
    glUseProgram(mvpPipeline.shaderProgram)

    # Setting up the clear screen color
    glClearColor(0.85, 0.85, 0.85, 1.0)

    # As we work in 3D, we need to check which part is in front,
    # and which one is at the back
    glEnable(GL_DEPTH_TEST)

    # Creating shapes on GPU memory
    cpuAxis = bs.createAxis(7)
    gpuAxis = es.GPUShape().initBuffers()
    mvpPipeline.setupVAO(gpuAxis)
    gpuAxis.fillBuffers(cpuAxis.vertices, cpuAxis.indices, GL_STATIC_DRAW)
    Scene = createScene(pipeline)
    Plane = createPlane(pipeline)
    # Using the same view and projection matrices in the whole application
    projection = tr.perspective(45, float(width)/float(height), 0.1, 100)

    glUseProgram(mvpPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    glUseProgram(pipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "projection"), 1, GL_TRUE, projection)

    viewPos = np.array([0,0, 0.5])
    view = tr.lookAt(
            viewPos,
            np.array([0,0,0]),
            np.array([0,1,0])
        )

    lightPosLocal = np.array([0.0, 0.0, 0.0])
    lightPosWorld = np.array([0, 10, 0,1])
    glUseProgram(mvpPipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "view"), 1, GL_TRUE, view)

    glUseProgram(pipeline.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(pipeline.shaderProgram, "view"), 1, GL_TRUE, view)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "viewPosition"), viewPos[0], viewPos[1], viewPos[2])

    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "La"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ld"), 2, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Kd"), 0.9, 0.9, 0.9)
    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(pipeline.shaderProgram, "lightPosition"), lightPosWorld[0], lightPosWorld[1], lightPosWorld[2])

    glUniform1ui(glGetUniformLocation(pipeline.shaderProgram, "shininess"), 100)
    glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "constantAttenuation"), 0.001)
    glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "linearAttenuation"), 0.1)
    glUniform1f(glGetUniformLocation(pipeline.shaderProgram, "quadraticAttenuation"), 0.01)


    glUseProgram(pipeline2.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(pipeline2.shaderProgram, "projection"), 1, GL_TRUE, projection)
    glUseProgram(pipeline2.shaderProgram)
    glUniformMatrix4fv(glGetUniformLocation(pipeline2.shaderProgram, "view"), 1, GL_TRUE, view)
    glUniform3f(glGetUniformLocation(pipeline2.shaderProgram, "viewPosition"), viewPos[0], viewPos[1], viewPos[2])

    glUniform3f(glGetUniformLocation(pipeline2.shaderProgram, "La"), 1.0, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(pipeline2.shaderProgram, "Ld"), 2, 1.0, 1.0)
    glUniform3f(glGetUniformLocation(pipeline2.shaderProgram, "Ls"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(pipeline2.shaderProgram, "Ka"), 0.2, 0.2, 0.2)
    glUniform3f(glGetUniformLocation(pipeline2.shaderProgram, "Kd"), 0.9, 0.9, 0.9)
    glUniform3f(glGetUniformLocation(pipeline2.shaderProgram, "Ks"), 1.0, 1.0, 1.0)

    glUniform3f(glGetUniformLocation(pipeline2.shaderProgram, "lightPosition"), lightPosWorld[0], lightPosWorld[1], lightPosWorld[2])

    glUniform1ui(glGetUniformLocation(pipeline2.shaderProgram, "shininess"), 100)
    glUniform1f(glGetUniformLocation(pipeline2.shaderProgram, "constantAttenuation"), 0.001)
    glUniform1f(glGetUniformLocation(pipeline2.shaderProgram, "linearAttenuation"), 0.1)
    glUniform1f(glGetUniformLocation(pipeline2.shaderProgram, "quadraticAttenuation"), 0.01)


    perfMonitor = pm.PerformanceMonitor(glfw.get_time(), 0.5)

    # glfw will swap buffers as soon as possible
    glfw.swap_interval(0)

    # Setting up the clear screen color
    glClearColor(0.56, 0.53, 0.53, 1.0)

    # initilize imgui context
    imgui.create_context()
    impl = GlfwRenderer(window)

    # Connecting the callback function 'on_key' to handle keyboard events
    # It is important to set the callback after the imgui setup
    glfw.set_key_callback(window, on_key)

    locationX = 0.0
    locationY = 0.0
    locationZ = 0.0
    locationRX = 0.0
    locationRY = 0.0
    locationRZ = 0.0

    while not glfw.window_should_close(window):

        impl.process_inputs()
        # Measuring performance
        perfMonitor.update(glfw.get_time())
        glfw.set_window_title(window, title + str(perfMonitor))
        # Using GLFW to check for input events

        glfw.set_mouse_button_callback(window, mouse_button_callback)
        glfw.set_cursor_pos_callback(window, cursor_position_callback)
        glfw.poll_events()

        # Clearing the screen in both, color and depth
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Filling or not the shapes depending on the controller state
        if (controller.fillPolygon):
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL)
        else:
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

        if controller.showAxis:
            glUseProgram(mvpPipeline.shaderProgram)
            glUniformMatrix4fv(glGetUniformLocation(mvpPipeline.shaderProgram, "model"), 1, GL_TRUE, tr.identity())
            mvpPipeline.drawCall(gpuAxis, GL_LINES)

        glUseProgram(pipeline.shaderProgram)

        x, y = glfw.get_cursor_pos(window)
        ray_direction = get_ray_direction(window, x, y)
        objRot = sg.findNode(Scene, "SceneNode")
        # glEnable(GL_DEPTH_TEST)
        sg.drawSceneGraphNode(Scene, pipeline, "model")
        glUseProgram(pipeline2.shaderProgram)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        sg.drawSceneGraphNode(Plane, pipeline2, "model")
        # imgui function
        impl.process_inputs()

        locationRX, locationRY, locationRZ= \
            transformGuiOverlay(locationRX, locationRY, locationRZ)

        impl.render(imgui.get_draw_data())

        # Once the render is done, buffers are swapped, showing only the complete scene.
        glfw.swap_buffers(window)

    # # freeing GPU memory
    # gpuQuad.clear()

    #freeing GPU memory
    gpuAxis.clear()
    Scene.clear()

    impl.shutdown()
    glfw.terminate()
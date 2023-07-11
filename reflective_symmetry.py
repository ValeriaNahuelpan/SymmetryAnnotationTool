from OpenGL.GL import *
import numpy as np
from itertools import combinations
import grafica.scene_graph as sg
import grafica.basic_shapes as bs
import grafica.easy_shaders as es
import grafica.transformations as tr
from grafica.assets_path import getAssetPath
from view_symmetry import *

def readOFF(filename, color):
    global vertices, faces
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

# stores the information of a reflective symmetry
# objects can have more than one reflective symmetry
class ReflectiveSymmetry:
    def __init__(self):
        self.pointsPairs = [] # list of tuples
        self.adding = False
        self.info = []
        self.colors = [
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


    # points to calculate reflective symmetry
    def create_point(self, pipeline, sceneNode, position, inv_matrix):
        sphereShape, _, _ = createOFFShape(pipeline, 'sphere.off', 0.5, 0.0, 0.0)
        if self.pointsPairs:
            lastPair = self.pointsPairs[-1]
            if len(lastPair) < 2:
                # second point in tuple
                print("ading point number 2 from pair number " + str(len(self.pointsPairs)))
                # add new node in scene
                p = sg.SceneGraphNode("point2-tuple" + str(len(self.pointsPairs)))
                p.transform = tr.uniformScale(0.00)
                p.childs += [sphereShape]
                # add a point in scene
                sceneNode.childs += [p]
                # we need the inverse matrix because the axis also rotates along with the object
                p.transform = tr.matmul([inv_matrix, tr.translate(position[0], position[1], position[2]), tr.uniformScale(0.0018)])
                self.pointsPairs[-1] += (p,)
                # finish adding points to the tuple to continue with a new
                self.adding = False
            else:
                # first point in tuple
                p = sg.SceneGraphNode("point1-tuple" + str(1+len(self.pointsPairs)))
                p.transform = tr.uniformScale(0.00)
                p.childs += [sphereShape]
                sceneNode.childs += [p]
                print("ading point number 1 from pair number " + str(1+len(self.pointsPairs)))
                p.transform = tr.matmul([inv_matrix, tr.translate(position[0], position[1], position[2]), tr.uniformScale(0.0018)])
                self.pointsPairs.append((p,))

        else:
            print("ading point number 1 from pair number 1 ")
            p = sg.SceneGraphNode("point1-tuple0")
            p.transform = tr.uniformScale(0.00)
            p.childs += [sphereShape]
            sceneNode.childs += [p]
            p.transform = tr.matmul([inv_matrix, tr.translate(position[0], position[1], position[2]), tr.uniformScale(0.0018)])
            self.pointsPairs.append((p,))

    def save_symmetry(self, midPoint, normal):
        info = {
                "point": midPoint,
                "normal": normal
               }
        self.info.append(info)
        return info



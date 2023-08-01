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


# stores the information of a rotational symmetry
class RotationalSymmetry:
    def __init__(self, maxPoints):
        self.adding = False
        self.axisAdded = False
        self.points = np.array([])
        self.maxPoints = maxPoints
        self.info = []


    # points to calculate rotational symmetry
    def create_point(self, pipeline, sceneNode, position, inv_matrix):
        if len(self.points) < (self.maxPoints):
            print("ading point" + str(len(self.points)))
            sphereShape, _, _ = createOFFShape(pipeline, 'sphere.off', 0.5, 0.0, 0.0)
            # add new node in scene
            p = sg.SceneGraphNode("point" + str(len(self.points)))
            p.transform = tr.uniformScale(0.00)
            p.childs += [sphereShape]
            # add a point in scene
            sceneNode.childs += [p]
            # we need the inverse matrix because the axis also rotates along with the object
            p.transform = tr.matmul([inv_matrix, tr.translate(position[0], position[1], position[2]), tr.uniformScale(0.0018)])
            self.points = np.append(self.points, p)

    # calculate the best plane and draw rotatation axis
    def calculate_plane(self, pipeline, sceneNode):
        if self.maxPoints > 2:
            cubeShape, _, _ = createOFFShape(pipeline, 'cube.off', 0.0, 0.0, 0.0)
            sphereShape, _, _ = createOFFShape(pipeline, 'sphere.off', 0.5, 0.0, 0.0)
            indices = np.arange(len(self.points))
            combinationsList = list(combinations(indices, 3)) # [(0,1,2),(0,1,3),(0,1,4)...]

            shorterDistance = 1e1000
            pointsBestPlane = [0,1,2]
            for combination in combinationsList:
                # position of points
                p = self.points[combination[0]].transform[:,3]
                q = self.points[combination[1]].transform[:,3]
                r = self.points[combination[2]].transform[:,3]
                # calculate plane formed by p, q and r
                mid_point =  ((p[0] + q[0] + r[0])/3, (p[1] + q[1] + r[1])/3, (p[2] + q[2] + r[2])/3)
                vector_1 = ((q[0] - p[0]), (q[1] - p[1]), (q[2] - p[2]))
                vector_2 = ((r[0] - p[0]), (r[1] - p[1]), (r[2] - p[2]))
                normal = ((vector_1[1]*vector_2[2] - vector_1[2]*vector_2[1]),
                          (vector_1[2]*vector_2[0] - vector_1[0]*vector_2[2]),
                          (vector_1[0]*vector_2[1] - vector_1[1]*vector_2[0]))
                magnitud = (normal[0]**2 + normal[1]**2 + normal[2]**2)**0.5
                normalizado = (normal[0] / magnitud, normal[1] / magnitud, normal[2] / magnitud)

                # measure the distance of all points to the plane
                totalDistances = 0
                for point in self.points:
                    if point not in [self.points[combination[0]],self.points[combination[1]],self.points[combination[2]]]:
                        pointPosition = point.transform[:,3]
                        v = np.array([pointPosition[0] - mid_point[0],pointPosition[1]-mid_point[1],pointPosition[2]-mid_point[2]])
                        distance = abs(np.dot(v,np.array(normalizado)))
                        totalDistances += distance

                # save the shorter distance, and the best plane at the moment
                if totalDistances < shorterDistance:
                    shorterDistance = totalDistances
                    pointsBestPlane = combination

            # to draw axis for the best plane
            p = self.points[pointsBestPlane[0]].transform[:,3]
            q = self.points[pointsBestPlane[1]].transform[:,3]
            r = self.points[pointsBestPlane[2]].transform[:,3]
            # calculate plane formed by p, q and r

            vector_1 = ((q[0] - p[0]), (q[1] - p[1]), (q[2] - p[2]))
            vector_2 = ((r[0] - p[0]), (r[1] - p[1]), (r[2] - p[2]))
            normal = ((vector_1[1]*vector_2[2] - vector_1[2]*vector_2[1]),
                        (vector_1[2]*vector_2[0] - vector_1[0]*vector_2[2]),
                        (vector_1[0]*vector_2[1] - vector_1[1]*vector_2[0]))
            magnitud = (normal[0]**2 + normal[1]**2 + normal[2]**2)**0.5
            normalizado = [normal[0] / magnitud, normal[1] / magnitud, normal[2] / magnitud]
            # calculate the mid point of all points
            x = 0
            y = 0
            z = 0
            for point in self.points:
                x += point.transform[:,3][0]
                y += point.transform[:,3][1]
                z += point.transform[:,3][2]
            mid_point =  [x/len(self.points), y/len(self.points), z/len(self.points)]
            # we use the class SymmetryPlane to create the plane to draw the normal to this
            symmetryPlane = SymmetryPlane(np.array(mid_point), np.array(normalizado))
            rotationAxis = sg.SceneGraphNode("axis")
            rotationAxis.childs += [cubeShape]
            sceneNode.childs += [rotationAxis]
            rotationAxis.transform = tr.matmul([symmetryPlane.transform, tr.scale(5,0.001,0.001)])
            info = {
                          "point": mid_point,
                          "normal": normalizado,
                          "pointRefined": [],
                          "normalRefined": [],
                          "isRef":False
                        }
            self.info.append(info)
            return self.info

    def drawAxis(self, pipeline, sceneNode):
        cubeShape, _, _ = createOFFShape(pipeline, 'cube.off', 0.0, 0.0, 0.0)
        symmetryPlane = SymmetryPlane(np.array(self.info[0]["point"]), np.array(self.info[0]["normal"]))
        rotationAxis = sg.SceneGraphNode("axis")
        rotationAxis.childs += [cubeShape]
        sceneNode.childs += [rotationAxis]
        rotationAxis.transform = tr.matmul([symmetryPlane.transform, tr.scale(5,0.001,0.001)])

    def save_symmetry(self, midPoint, normal, pointRefined=[], normalRefined=[], isRef=False):
        info = {
                "point": midPoint,
                "normal": normal,
                "pointRefined": pointRefined,
                "normalRefined": normalRefined,
                "isRef": isRef
               }
        self.info.append(info)
        return info









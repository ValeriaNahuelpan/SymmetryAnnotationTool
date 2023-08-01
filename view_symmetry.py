import argparse
import os
import polyscope as ps
import numpy as np
import json
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

def chamfer_distance(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """

    if direction=='y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction=='x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction=='bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")

    return chamfer_dist

def matmul(mats):
    out = mats[0]

    for i in range(1, len(mats)):
        out = np.matmul(out, mats[i])

    return out

#Returns the rotation matrix in X
def rotationX(theta):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    return np.array([
        [1,0,0,0],
        [0,cos_theta,-sin_theta,0],
        [0,sin_theta,cos_theta,0],
        [0,0,0,1]], dtype = np.float32)

#Returns the rotation matrix in Y
def rotationY(theta):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    return np.array([
        [cos_theta,0,sin_theta,0],
        [0,1,0,0],
        [-sin_theta,0,cos_theta,0],
        [0,0,0,1]], dtype = np.float32)

#Returns the rotation matrix in Y
def rotationZ(theta):
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)

    return np.array([
        [cos_theta,-sin_theta,0,0],
        [sin_theta,cos_theta,0,0],
        [0,0,1,0],
        [0,0,0,1]], dtype = np.float32)

#Returns the translation matrix
def translate(tx, ty, tz):
    return np.array([
        [1,0,0,tx],
        [0,1,0,ty],
        [0,0,1,tz],
        [0,0,0,1]], dtype = np.float32)

def rotationA(theta, axis):
    s = np.sin(theta)
    c = np.cos(theta)

    assert axis.shape == (3,)

    x = axis[0]
    y = axis[1]
    z = axis[2]

    return np.array([
        # First row
        [c + (1 - c) * x * x,
        (1 - c) * x * y - s * z,
        (1 - c) * x * z + s * y,
        0],
        # Second row
        [(1 - c) * x * y + s * z,
        c + (1 - c) * y * y,
        (1 - c) * y * z - s * x,
        0],
        # Third row
        [(1 - c) * x * z - s * y,
        (1 - c) * y * z + s * x,
        c + (1 - c) * z * z,
        0],
        # Fourth row
        [0,0,0,1]], dtype = np.float32)

def generateRotationTransform(point, normal, angle):
    return matmul([translate(point[0], point[1], point[2]), rotationA(angle, normal), translate(-point[0], -point[1], -point[2])])

def generateSymmetryTransform(point, normal):
    T = translate(-point[0], -point[1], -point[2])
    Tinv = translate(point[0], point[1], point[2])

    a = normal[0]
    b = normal[1]
    c = normal[2]

    Ref = np.array([
        [1-2*a**2, -2*a*b, -2*a*c, 0],
        [-2*a*b, 1-2*b**2, -2*b*c, 0],
        [-2*a*c, -2*b*c, 1-2*c**2, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    return matmul([Tinv, Ref, T])

def refineReflectionTransform(point, normal, points):
    
    transf = generateSymmetryTransform(point, normal)
    points2=transformPoints(points, transf)

    dist = chamfer_distance(points, points2, metric='l2', direction='bi')

    print(f'Distance: {dist}')

    for i in range(20):
        delta = 0.01*(np.random.rand(3)-0.5)
        normal2 = normal + delta
        normal2 = normal2 / np.linalg.norm(normal2)

        transf = generateSymmetryTransform(point, normal2)
        ones = np.ones((1,points.shape[0]))
        points2 = np.concatenate((points.T, ones))

        points2 = transf@points2
        points2 = points2[0:3,:].T

        dist2 = chamfer_distance(points, points2, metric='l2', direction='bi')

        if dist2 < dist:
            dist = dist2
            normal = normal2
            print(f'New distance: {dist}')
  
    return normal

def transformPoints(points, transf):
    ones = np.ones((1,points.shape[0]))
    points2 = np.concatenate((points.T, ones))

    points2 = transf@points2
    points2 = points2[0:3,:].T

    return points2

def refineRotationTransform(point, normal, points):
    transf = [generateRotationTransform(point, normal, x) for x in np.linspace(0, 2*np.pi, 6)]
    P = [transformPoints(points, x) for x in transf]
    dist = [chamfer_distance(points, x, metric='l2', direction='bi') for x in P]
    dist = sum(dist)

    print(f'Distance: {dist}')

    for epoch in range(5):
        for i in range(10):
            delta = 0.01*(np.random.rand(3)-0.5)
            normal2 = normal + delta
            normal2 = normal2 / np.linalg.norm(normal2)

            transf = [generateRotationTransform(point, normal2, x) for x in np.linspace(0, 2*np.pi, 6)]
            P = [transformPoints(points, x) for x in transf]
            dist2 = [chamfer_distance(points, x, metric='l2', direction='bi') for x in P]
            dist2 = sum(dist2)

            if dist2 < dist:
                dist = dist2
                normal = normal2
                print(f'New distance: {dist}')

        for i in range(10):
            delta = 0.01*(np.random.rand(3)-0.5)
            d = np.dot(delta, normal)
            projected_point = point - d*normal

            transf = [generateRotationTransform(projected_point, normal, x) for x in np.linspace(0, 2*np.pi, 6)]
            P = [transformPoints(points, x) for x in transf]
            dist2 = [chamfer_distance(points, x, metric='l2', direction='bi') for x in P]
            dist2 = sum(dist2)

            if dist2 < dist:
                dist = dist2
                point = projected_point
                print(f'New distance: {dist}')

    return point, normal

#Stores the information of a symmetry plane
class SymmetryPlane:
    def __init__(self, point, normal):
        #3D coords of a canonical plane (for drawing)
        self.coordsBase = np.array([[0,-1,-1],[0,1,-1],[0,1,1],[0,-1,1]], dtype=np.float32)
        #Indices for the canonical plane
        self.trianglesBase = np.array([[0,1,3],[3,1,2]], dtype=np.int32)

        #The plane is determined by a normal vector and a point
        self.point = point.astype(np.float32)
        self.normal = normal
        self.normal = self.normal / np.linalg.norm(self.normal)

        self.compute_geometry()

    #Applies a rotation to the plane
    def apply_rotation(self, rot):
        transf = rot.copy()
        transf = transf[0:3,0:3]
        transf = np.linalg.inv(transf).T

        self.normal = transf@self.normal

        self.compute_geometry()

    def apply_traslation(self, x, y, z):
        self.point[0] = self.point[0] + x
        self.point[1] = self.point[1] + y
        self.point[2] = self.point[2] + z

        #print(self.point)

    #Transforms the canonical plane to be oriented wrt the normal
    def compute_geometry(self):
        #Be sure the vector is normal
        self.normal = self.normal / np.linalg.norm(self.normal)
        #print(f'First normal: {self.normal}')
        a, b, c = self.normal

        h = np.sqrt(a**2 + c**2)

        if h < 0.0000001:
            angle = np.pi/2

            T = translate(self.point[0], self.point[1], self.point[2])
            Rz = rotationZ(angle)
            transform = matmul([T, Rz])
        else:

            Rzinv = np.array([
                [h, -b, 0, 0],
                [b, h, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)

            Ryinv = np.array([
                [a/h, 0, -c/h, 0],
                [0, 1, 0, 0],
                [c/h, 0, a/h, 0],
                [0, 0, 0, 1]
            ], dtype=np.float32)

            T = translate(self.point[0], self.point[1], self.point[2])

            transform = matmul([T, Ryinv, Rzinv])
        self.transform = transform
        ones = np.ones((1,4))
        self.coords = np.concatenate((self.coordsBase.T, ones))

        self.coords = transform@self.coords
        #print(f'Transform:\n{transform}')

        self.coords = self.coords[0:3,:].T
        #print(f'Coord:\n{self.coords}')

# #Recibimos un json con la informacion de las simetrias
# parser = argparse.ArgumentParser()
# parser.add_argument('--input', type=str, default='', help='')
# opt = parser.parse_args()

# with open('symmetries.json') as f:
#     data = json.load(f)

# print(data)

#Read triangle mesh with open3d
# mesh = o3d.io.read_triangle_mesh(data['objects'][0]['file'])

# #Convert mesh to numpy arrays
# points = np.asarray(mesh.vertices)
# triangles = np.asarray(mesh.triangles)


#Refinamos las simetrias rotacionales
# new_point, new_rot = refineRotationTransform(symmetry_list_rot[0].point, symmetry_list_rot[0].normal, points)
# transf = generateRotationTransform(new_point, new_rot/np.linalg.norm(new_rot), np.pi/2)
# points3 = transformPoints(points, transf)

#Creamos un nuevo plano con la simetría rotacional refinada
#sym_rot = SymmetryPlane(point=new_point, normal=new_rot)


#ps.register_curve_network("rot", np.array([-sym_rot.normal+sym_rot.point, sym_rot.normal+sym_rot.point]), np.array([[0,1]]), radius=0.002)



#for i, sym2 in enumerate(symmetry_list_rot):
#    ps.register_curve_network("rot_"+str(i), np.array([-sym2.normal+sym2.point, sym2.normal+sym2.point]), np.array([[0,1]]), radius=0.002)

#Leemos las simetrias rotacionales
# symmetry_list_rot = []
# for i in range(len(data['objects'][0]['symmetries']['rotational'])):
#     sym = data['objects'][0]['symmetries']['rotational'][i]
#     #print(sym['point'])
#     #print(sym['normal'])
#     symmetry_list_rot.append(SymmetryPlane(point=np.array(sym['point']), normal=np.array(sym['normal'])))

# #Leemos las simetrias reflectivas
# symmetry_list_ref = []
# for i in range(len(data['objects'][0]['symmetries']['reflectives'])):
#     sym = data['objects'][0]['symmetries']['reflectives'][i]
#     #print(sym['point'])
#     #print(sym['normal'])
#     symmetry_list_ref.append(SymmetryPlane(point=np.array(sym['point']), normal=np.array(sym['normal'])))
# #Refinamos las simetrias reflectiva
# new_normal = refineReflectionTransform(symmetry_list_ref[0].point, symmetry_list_ref[0].normal, points)
# transf = generateSymmetryTransform(symmetry_list_ref[0].point, new_normal/np.linalg.norm(new_normal))
# points2 = transformPoints(points, transf)

# #Creamos un nuevo plano con la simetría reflectiva refinada
# sym_ref = SymmetryPlane(point=symmetry_list_ref[0].point, normal=new_normal)

# #Visualizamos en Polyscope
# ps.init()
# ps.register_surface_mesh("mesh1", points, triangles)
# ps.register_surface_mesh("mesh2", points2, triangles)
# mesh = ps.register_surface_mesh("final", sym_ref.coords, sym_ref.trianglesBase)
# mesh.set_transparency(0.8)
# for i, sym2 in enumerate(symmetry_list_ref):
#     mesh2 = ps.register_surface_mesh("sym_"+str(i), sym2.coords, sym2.trianglesBase)
#     mesh2.set_transparency(0.8)
# ps.show()
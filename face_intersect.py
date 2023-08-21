"""
from sympy import Point3D, Line3D, Plane
import math
import numpy as np

x_theta = y_theta = z_theta = 0
print(x_theta, y_theta, z_theta)

print(-5 % -2)

t = -15 * math.pi / 16
t = math.tan(t)
t = math.atan(t)
print(t / math.pi)

a = None
a = np.array([1, 2])
print(a)
# class Face_Intersect:
    # def __init__(self, )

# def 

m1 = np.array([2, 1, 3, 4])
# for i in range(j = 3, step = 2):
#     print(m1[i])
m1[[0, 2]] = m1[[2, 0]]
print(m1 - 1)
m2 = np.array([[7, 5], [6, 8]])
m2[0] = [1, 2]
# m3 = m1 @ m2
print(m2)

vertices = np.array([
    (-50, -50, -50, 1), (-50, 50, -50, 1), # Vertices 0, 1
    (50, 50, -50, 1), (50, -50, -50, 1), # Vertices 2, 3
    (-50, -50, 50, 1), (-50, 50, 50, 1), # Vertices 4, 5
    (50, 50, 50, 1), (50, -50, 50, 1) # Vertices 6, 7
])

z_vertices = np.array([vertex[2] for vertex in vertices])
print(max(z_vertices))


faces = np.array([
    (0, 1, 2, 3), (4, 5, 6, 7), # Faces 0, 1
    (0, 4, 5, 1), (2, 3, 7, 6), # Faces 2, 3
    (1, 2, 6, 5), (0, 3, 7, 4) # Faces 4, 5
])

vertex_faces = np.zeros(shape = (6, 4, 3))

for i in range(len(faces)):
    for x in range(len(faces[i])):
        vertex_faces[i][x] = vertices[faces[i][x]][:3]
        # print(type(vertices[faces[i][x]]))

print(vertex_faces)

opposite_vertices = np.array([
    (0, 6), (6, 0), (1, 7), (7, 1),
    (2, 4), (4, 2), (3, 5), (5, 3)
])


a = Plane(Point3D(1, 2, 3), normal_vector=(1, 1, 1))
b = Point3D(0, 0, 0)
pt = Point3D(1, 1, 1)
print(list((b + pt)/2))
# print(a.intersection(b))
c = Line3D(Point3D(1, 4, 7), Point3D(2, 2, 2))
pt = a.intersection(c)
d = Plane(Point3D(6, 0, 0), normal_vector=(2, -5, 3))
e = Plane(Point3D(2, 0, 0), normal_vector=(3, 4, -3))
print(d.intersection(e))
pt = np.asarray(pt)
print(pt, pt.shape)

# List of all faces with vertices

"""

from collections import Counter
import numpy as np

arr1 = np.array([1, 2, 3, 4, -5, 6])
print(np.argsort(arr1)[-2], arr1)

r2d2 = np.array([[1, 2, 3], [1, 2, 3], [4, 5, 6],])
print('test', np.intersect1d(np.where(np.isin(r2d2, arr1))[0], np.arange(len(arr1))))

tan = np.arctan2(0, 0)
print(tan)
print(tan == -np.pi / 4)

def change_parameter(a):
    a -= 5
    return a

para = 5
print(change_parameter(para), para)
arr = np.array([1, 2, 3])
print(*arr)

vec_a = np.array([1, 2, 3])
print('vec', arr == vec_a)
vec_b = np.array([4, 5, 6])
c = np.square(vec_b)
c = np.sum(c)
print(c)
vec_a = vec_a[:2]
vec_b = vec_b[:2]
dot = np.dot(vec_a, vec_b)
print(vec_b - vec_a)
print(dot)

vertices = np.array([
            (-6, -3, -4, 1), (-6, 3, -4, 1), # Vertices 0, 1
            (6, 3, -4, 1), (6, -3, -4, 1), # Vertices 2, 3
            (-6, -3, 4, 1), (-6, 3, 4, 1), # Vertices 4, 5
            (6, 3, 4, 1), (6, -3, 4, 1) # Vertices 6, 7
        ])
faces = np.array([
            (0, 1, 2, 3), (4, 5, 6, 7), # Faces 0, 1
            (0, 4, 5, 1), (2, 3, 7, 6), # Faces 2, 3
            (1, 2, 6, 5), (0, 3, 7, 4) # Faces 4, 5
        ])

for face in faces:
    print(vertices[face])

arr = np.array([])
for i in range(16):
    arr = np.append(arr, i)
arr = np.reshape(arr, (-1, 4, 2))
print(arr)

a = np.array([1, 2, 3, 0])
b = [[item, count] for item, count in Counter(a).items() if count > 1]
print(b)
list1 = np.array([1, 3, 4, 5])
list2 = np.array([6, 7, 8, 9])
arr = np.unique(list1, return_index=True, return_counts=True, return_inverse=True)
print(arr)
# print(list1[1, 2, 3])

common_elements = list(set(list2).intersection(list1))

if(not common_elements):
    print(True, 5)
else:
    print(False)

arr2 = np.array([], dtype=object)
arr2 = np.append(arr2, [[1,1,1]])
arr2 = np.append(arr2, [[2,2,2]])
arr2 = np.reshape(arr2, (-1, 3))
print(arr2)

vertices = np.array([(2, 2, 2), (-2, -2, -2)])
for p1, p2, p3 in vertices:
    print(p1, p2, p3)

cam_pos = np.array([-2.5, 2, 1])

vector = np.array([
  cam_pos, vertices[1]
])

delta_x = vector[1][0] - vector[0][0]
delta_y = vector[1][1] - vector[0][1]
delta_z = vector[1][2] - vector[0][2]

scalar_x = (vertices[0][0] - cam_pos[0]) / delta_x
scalar_y = (vertices[0][1] - cam_pos[1]) / delta_y
scalar_z = (vertices[0][2] - cam_pos[2]) / delta_x

x_intersect = np.array([cam_pos[1] + delta_y * scalar_x, cam_pos[2] + delta_z * scalar_x])
y_intersect = np.array([cam_pos[0] + delta_x * scalar_y, cam_pos[2] + delta_z * scalar_y])
z_intersect = np.array([cam_pos[0] + delta_x * scalar_z, cam_pos[1] + delta_y * scalar_z])

invisible = (
    (np.abs(x_intersect[0]) <= np.abs(vertices[0][1]) and np.abs(x_intersect[1]) <= np.abs(vertices[0][2])) or \
    (np.abs(y_intersect[0]) <= np.abs(vertices[0][0]) and np.abs(y_intersect[1]) <= np.abs(vertices[0][2])) or \
    (np.abs(z_intersect[0]) <= np.abs(vertices[0][0]) and np.abs(z_intersect[1]) <= np.abs(vertices[0][1]))
    )

print(invisible)

a = np.array([], dtype = object)
a = np.append(a, (1, 2, 3, 4))
print(a)


"""
if not invisible and x_intersect[1] <= np.abs(vertices[0][2]) and x_intersect[0] <= np.abs(vertices[0][1]):
    invisible = True

if not invisible:
  y_theta = np.arctan2(y_intersect[1], y_intersect[0])
  dist = np.sqrt(y_intersect[1]**2 + y_intersect[0]**2)
  h_y = dist * np.cos(y_theta)
  v_y = dist * np.sin(y_theta)
  if y_intersect[1] <= np.abs(vertices[0][2]) and y_intersect[0] <= np.abs(vertices[0][0]):
    invisible = True

if not invisible:
  z_theta = np.arctan2(z_intersect[1], z_intersect[0])
  dist = np.sqrt(z_intersect[1]**2 + z_intersect[0]**2)
  h_z = dist * np.cos(z_theta)
  v_z = dist * np.sin(z_theta)
  if v_z <= np.abs(vertices[0][1]) and h_z <= np.abs(vertices[0][0]):
    invisible = True
"""

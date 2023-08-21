import pygame as pg
import pygame.gfxdraw
# from collections import Counter
from matrix_functions import *

class Object3D:
    def __init__(self, render, vertices_arr):
        self.render = render
        self.vertices = vertices_arr
        
        self.faces = np.array([
            (0, 4, 5, 1), (2, 3, 7, 6), # Faces 2, 3
            (0, 1, 2, 3), (4, 5, 6, 7), # Faces 0, 1
            (1, 2, 6, 5), (0, 3, 7, 4) # Faces 4, 5
        ])
        
        self.visible_pts = np.array([])

        """
        Front Face: 0
        Back Face: 1
        Left Face: 2
        Right Face: 3
        Top Face: 4
        Bottom Face: 5
        """

        self.center = np.array([
            (self.vertices[0][0] + self.vertices[6][0]) / 2, 
            (self.vertices[0][1] + self.vertices[6][1]) / 2, 
            (self.vertices[0][2] + self.vertices[6][2]) / 2
        ])
        
        self.h_v_rad_to_center = np.array([
            np.pi / 2 - np.abs(np.arctan2(self.center[2], self.center[0])),
            np.arctan2(self.center[1], self.center[2]),
            self.center[2] >= 0
        ])

        # self.x_theta = self.y_theta = self.z_theta = 0
        
        # self.font = pg.font.SysFont('Arial', 30, bold = True)
        # self.color_faces = [(pg.Color('orange'), face) for face in self.faces]
        # self.movement_flag, self.draw_vertices = True, True
        # self.label = ''

    def draw(self, cam_pos):
        self.screen_projection(cam_pos)
        # self.movement()

    def screen_projection(self, cam_pos):
        vertices = self.vertices @ self.render.camera.camera_matrix()
        vertices = vertices @ self.render.projection.projection_matrix
        vertices /= vertices[:, -1].reshape(-1, 1)
        vertices[(vertices > 1000) | (vertices < -1000)] = 0
        vertices = vertices @ self.render.projection.to_screen_matrix
        vertices = vertices[:, :2]

        face_colors = [pg.Color('red'), pg.Color('blue'), 
                        pg.Color('chartreuse'), pg.Color('darkviolet'), 
                        pg.Color('cornsilk'), pg.Color('gold'), 
                        pg.Color('aqua'), pg.Color('aliceblue')]

        visible_faces = self.back_face_culling(cam_pos)
        self.visible_pts = visible_faces[0]
        self.visible_pts_indices = visible_faces[1]

        for index, face in enumerate(self.visible_pts): 
            polygon = vertices[face]
            if ((not np.any((polygon == self.render.H_WIDTH) | (polygon == self.render.H_HEIGHT)))):
                pygame.gfxdraw.filled_polygon(self.render.screen, polygon, face_colors[self.visible_pts_indices[index]])
    
    def back_face_culling(self, cam_pos):
        visible_faces = np.array([])
        visible_faces_indices = np.array([])
        cam_pos = cam_pos[:3]

        for index, face in enumerate(self.faces):
            vertices = self.vertices[face]
            origin = vertices[1][:3]
            p = vertices[0][:3]
            q = vertices[2][:3]

            vector_p = p - origin # x-axis
            vector_q = q - origin # y-axis
            normal_vector = np.cross(vector_p, vector_q)
            face_center = np.array([(p[i] + q[i]) / 2 for i in range(3)])
            center_vector = self.center - face_center

            mag_norm_vec = np.sqrt(np.sum(np.square(normal_vector)))
            mag_cent_vec = np.sqrt(np.sum(np.square(center_vector)))
            center_dot = np.dot(normal_vector, center_vector)
            cos_theta = center_dot / (mag_cent_vec * mag_norm_vec)

            if int(cos_theta) == 1:
                normal_vector *= -1

            cam_vec = origin - cam_pos
            cam_dot_prod = np.dot(cam_vec, normal_vector)

            if cam_dot_prod < 0:
                visible_faces = np.append(visible_faces, face)
                visible_faces_indices = np.append(visible_faces_indices, index)

        visible_faces = visible_faces.astype(int)
        visible_faces_indices = visible_faces_indices.astype(int)
        visible_faces = np.reshape(visible_faces, (-1, 4))

        return [visible_faces, visible_faces_indices]

    def translate_obj(self, pos):
        self.vertices = self.vertices @ translate(pos)
        self.update_center()
    
    def scale_obj(self, n):
        self.vertices = self.vertices @ scale(n)
        self.update_center()
    
    def rotate_x_obj(self, theta):
        self.vertices = self.vertices @ rotate_x(theta)
        self.update_center()     
        
    def rotate_y_obj(self, theta):
        self.vertices = self.vertices @ rotate_y(theta)
        self.update_center()
        
    def rotate_z_obj(self, theta):
        self.vertices = self.vertices @ rotate_z(theta)
        self.update_center()
        
    def update_center(self):
        self.center = np.array([
            (self.vertices[0][0] + self.vertices[6][0]) / 2, 
            (self.vertices[0][1] + self.vertices[6][1]) / 2, 
            (self.vertices[0][2] + self.vertices[6][2]) / 2
        ])
        
        z_max_coord = max(np.array([vertex[2] for vertex in self.vertices]))

        self.h_v_rad_to_center = np.array([
            np.pi / 2 - np.abs(np.arctan2(self.center[2], self.center[0])),
            np.arctan2(self.center[1], self.center[2]),
            self.center[2] >= 0,
            (self.center[2] + z_max_coord) / 2 >= 0
        ])

"""
class Axes(Object3D):
    def __init__(self, render):
        super().__init__(render)
        self.vertices = np.array([
            (0, 0, 0, 1), (1, 0, 0, 1),
            (0, 1, 0, 1), (0, 0, 1, 1)
        ])
        self.faces = np.array([
            (0, 1), (0, 2), (0, 3)
        ])
        self.colors = [pg.Color('red'), pg.Color('green'), pg.Color('blue')]
        self.color_faces = [(color, face) for color, face in zip(self.colors, self.faces)]
        self.draw_vertices = False
        self.label = 'XYZ'
"""
# Unused Methods

"""
def movement(self):
    if self.movement_flag:
        self.rotate_y_obj(pg.time.get_ticks() % 0.005)

def adjust_coords(self, coordinate):
    new_coordinate = coordinate[:3]
    new_coordinate[[1, 2]] = new_coordinate[[2, 1]]
    # print('coord', coordinate)
    return new_coordinate

def is_coord_invisible(self, cam_pos, p1, p2):
    vertices = np.array([p1, p2])

    vector = np.array([cam_pos, p2])
    # print('scalar', vector[1])

    delta_x = vector[1][0] - vector[0][0]
    delta_y = vector[1][1] - vector[0][1]
    delta_z = vector[1][2] - vector[0][2]

    scalar_x = (vertices[0][0] - cam_pos[0]) / delta_x
    scalar_y = (vertices[0][1] - cam_pos[1]) / delta_y
    scalar_z = (vertices[0][2] - cam_pos[2]) / delta_z

    x_intersect = np.array([
        p1[0],
        cam_pos[1] + delta_y * scalar_x, 
        cam_pos[2] + delta_z * scalar_x
    ])
    y_intersect = np.array([
        cam_pos[0] + delta_x * scalar_y,
        p1[1],
        cam_pos[2] + delta_z * scalar_y
    ])
    z_intersect = np.array([
        cam_pos[0] + delta_x * scalar_z,
        cam_pos[1] + delta_y * scalar_z,
        p1[2]
    ])

    invisible = (
        (np.abs(x_intersect[1]) < np.abs(vertices[0][1]) and np.abs(x_intersect[2]) < np.abs(vertices[0][2])) or \
        (np.abs(y_intersect[0]) < np.abs(vertices[0][0]) and np.abs(y_intersect[2]) < np.abs(vertices[0][2])) or \
        (np.abs(z_intersect[0]) < np.abs(vertices[0][0]) and np.abs(z_intersect[1]) < np.abs(vertices[0][1]))
    )

    return [invisible, p2]

def farthest_coords(self, cam_pos):
    max_dist = 0
    farthest_pt = 0
    for i, vertex in enumerate(self.vertices):
        delta_x = np.square(cam_pos[0] - vertex[0])
        delta_y = np.square(cam_pos[1] - vertex[1])
        delta_z = np.square(cam_pos[2] - vertex[2])
        dist = np.sqrt(delta_x + delta_y + delta_z)
        if(dist > max_dist):
            max_dist = dist
            farthest_pt = i
    return farthest_pt

def invisible_coords(self, cam_pos):
    invisible_pts = np.array([])

    original_vertices = self.vertices
    center = self.center

    original_vertices = original_vertices @ translate(-center[:3])
    original_vertices = original_vertices @ rotate_x(-self.x_theta)
    original_vertices = original_vertices @ rotate_y(-self.y_theta)
    original_vertices = original_vertices @ rotate_z(-self.z_theta)
    # print('og_verts', original_vertices)

    # new_vertices = np.empty(shape=(8, 2), dtype=object)
    # new_vertices = np.reshape(new_vertices, (8, 2, 3))

    # for i in range(len(original_vertices)):
    #     new_vertices[i][i % 2] = self.adjust_coords(original_vertices[i])
    
    new_vertices = [self.adjust_coords(vertex) for vertex in original_vertices]
    # print('new_verts', new_vertices)
    # print('invisible coords', np.shape(new_vertices))
    opposite_vertices = np.array([
        (new_vertices[6], new_vertices[0]), 
        (new_vertices[7], new_vertices[1]),
        (new_vertices[4], new_vertices[2]), 
        (new_vertices[5], new_vertices[3]),
        (new_vertices[2], new_vertices[4]), 
        (new_vertices[3], new_vertices[5]), 
        (new_vertices[0], new_vertices[6]), 
        (new_vertices[1], new_vertices[7]), 
    ])

    cam_pos = cam_pos @ translate(-self.center[:3])
    cam_pos = cam_pos @ rotate_x(-self.x_theta)
    cam_pos = cam_pos @ rotate_y(-self.y_theta)
    cam_pos = cam_pos @ rotate_z(-self.z_theta)

    cam_pos = self.adjust_coords(cam_pos)

    for i, vertex_pair in enumerate(opposite_vertices):
        # print('pair', vertex_pair)
        vertex = self.is_coord_invisible(cam_pos, vertex_pair[0], vertex_pair[1])
        # print('vertex_loop', vertex)
        if vertex[0]:
            invisible_pts = np.append(invisible_pts, i)

    return invisible_pts

def plane_3D_to_2D(self, cam_pos):
    cam_pos = cam_pos[:3]
    origin = self.closest_vertices[1][:3]
    p = self.closest_vertices[0][:3]
    q = self.closest_vertices[2][:3]
    vector_p = p - origin # x-axis
    vector_q = q - origin # y-axis
    normal_vector = np.cross(vector_p, vector_q)
    plane_eqn = np.append(normal_vector, np.dot(normal_vector, origin))
    vectors = np.array([])
    for vertex in self.farthest_vertices:
        for i in range(3):
            vectors = np.append(vectors, vertex[i] - cam_pos[i])
    vectors = np.reshape(vectors, (4, 3))
    intersection_pts = np.array([])
    for vector in vectors:
        close_vector = origin - cam_pos
        dot_1 = np.dot(normal_vector, close_vector)
        dot_2 = np.dot(normal_vector, vector)
        scalar = dot_1 / dot_2
        temp_sum = 0
        intersection_pt = np.array([])
        for i in range(3):
            intersection_pt = np.append(intersection_pt, cam_pos[i] + scalar * vector[i])
            intersection_pts = np.append(intersection_pts, cam_pos[i] + scalar * vector[i])
            temp_sum += plane_eqn[i] * (cam_pos[i] + scalar * vector[i])
        if(np.abs(temp_sum - plane_eqn[3]) >= 1):
            raise ValueError(f'Point is not on plane: {plane_eqn[0]}x + {plane_eqn[1]}y + {plane_eqn[2]}z = {plane_eqn[3]}; Current point is {intersection_pt}; Current sum is {temp_sum} which is {plane_eqn[3] - temp_sum} away')
        
    intersection_pts = np.reshape(intersection_pts, (4, 3))
    
    theta_3D = np.abs(np.arctan2(vector_p[2], vector_p[0]))
    dist_o_p = np.sqrt(np.sum(np.square(vector_p)))
    dist_o_q = np.sqrt(np.sum(np.square(vector_q)))
    x_axis_vector_2D = y_axis_vector_2D = np.array([])
    x_axis_vector_3D  = np.array([])
    if theta_3D <= np.pi / 4:
        x_axis_vector_2D = np.append(x_axis_vector_2D, [np.copysign(dist_o_p, vector_p[0]), 0])
        y_axis_vector_2D = np.append(y_axis_vector_2D, [0, np.copysign(dist_o_q, vector_q[2])])
        x_axis_vector_3D = vector_p
        # y_axis_vector_3D = vector_q
    else:
        x_axis_vector_2D = np.append(x_axis_vector_2D, [np.copysign(dist_o_q, vector_q[0]), 0])
        y_axis_vector_2D = np.append(y_axis_vector_2D, [0, np.copysign(dist_o_p, vector_p[2])])
        x_axis_vector_3D = vector_q
        # y_axis_vector_3D = vector_p
    invisible_pts = np.array([])
    for point in intersection_pts:
        ext_vector = point - origin
        mag_ext = np.sqrt(np.sum(np.square(ext_vector)))
        mag_x_axis_vector_2D = np.abs(x_axis_vector_2D[0])
        numerator = np.dot(ext_vector, x_axis_vector_3D)
        denominator = mag_ext * mag_x_axis_vector_2D
        theta_2D = np.arccos(numerator / denominator)
        x_2D = np.cos(theta_2D) * mag_ext
        y_2D = np.sin(theta_2D) * mag_ext
        invisible_x = invisible_y = False
        if x_axis_vector_2D[0] < 0:
            invisible_x = x_2D >= x_axis_vector_2D[0] and x_2D <= 0
        else:
            invisible_x = x_2D <= x_axis_vector_2D[0] and x_2D >= 0
        if y_axis_vector_2D[1] < 0:
            invisible_y = y_2D >= y_axis_vector_2D[1] and y_2D <= 0
        else:
            invisible_y = y_2D <= y_axis_vector_2D[1] and y_2D >= 0
        if invisible_x and invisible_y:
            invisible_pts = np.append(invisible_pts, [*point, 1])
    invisible_pts = np.reshape(invisible_pts, (-1, 4))
    return invisible_pts
"""
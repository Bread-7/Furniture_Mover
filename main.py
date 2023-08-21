import pygame as pg
from object_3d import *
from camera import *
from projection import *
# import math

class SoftwareRender:
    def __init__(self):
        pg.init()
        self.RES = self.WIDTH, self.HEIGHT = 1280, 720
        self.H_WIDTH, self.H_HEIGHT = self.WIDTH // 2, self.HEIGHT // 2
        self.FPS = 60
        self.moving_speed = 0.5
        self.rotating_speed = 0.02
        self.screen = pg.display.set_mode(self.RES)
        self.clock = pg.time.Clock()
        self.create_objects()

    def create_objects(self):
        self.camera = Camera(self, np.array([0.001, 0.001, 0.001]))
        self.projection = Projection(self)
        
        self.object1 = Object3D(self, vertices_arr = np.array([
            (-6, -3, -4, 1), (-6, 3, -4, 1), # Vertices 0, 1
            (6, 3, -4, 1), (6, -3, -4, 1), # Vertices 2, 3
            (-6, -3, 4, 1), (-6, 3, 4, 1), # Vertices 4, 5
            (6, 3, 4, 1), (6, -3, 4, 1) # Vertices 6, 7
        ]))

        self.object = Object3D(self, vertices_arr = np.array([
            (2, 2 - 2, 2 + 15, 1), (2, 4, 2 + 15, 1), # Vertices 0, 1
            (6, 4, 2 + 15, 1), (6, 2 - 2, 2 + 15, 1), # Vertices 2, 3
            (2, 2 - 2, 6 + 15, 1), (2, 4, 6 + 15, 1), # Vertices 4, 5
            (6, 4, 6 + 15, 1), (6, 2 - 2, 6 + 15, 1) # Vertices 6, 7
        ]))

        # Cube
        # self.object = Object3D(self, vertices_arr = np.array([
        #     (-2, -2, 17, 1), (-2, 2, 17, 1), # Vertices 0, 1
        #     (2, 2, 17, 1), (2, -2, 17, 1), # Vertices 2, 3
        #     (-2, -2, 21, 1), (-2, 2, 21, 1), # Vertices 4, 5
        #     (2, 2, 21, 1), (2, -2, 21, 1) # Vertices 6, 7
        # ]))

        # self.axes = Axes(self)
        # self.axes.translate_obj([0.7, 0.9, 0.7])
        # self.world_axes = Axes(self)
        # self.world_axes.movement_flag = False
        # self.world_axes.scale_obj(2.5)
        # self.world_axes.translate_obj([0.0001, 0.0001, 0.0001]) 

    def config_key_bindings(self):
        key = pg.key.get_pressed()
        if key[pg.K_a]:
            self.object.translate_obj([-self.moving_speed, 0, 0])
            self.object1.translate_obj([-self.moving_speed, 0, 0])
        if key[pg.K_d]:
            self.object.translate_obj([self.moving_speed, 0, 0])
            self.object1.translate_obj([self.moving_speed, 0, 0])
        if key[pg.K_w]:
            self.object.translate_obj([0, 0, -self.moving_speed])
            self.object1.translate_obj([0, 0, -self.moving_speed])
        if key[pg.K_s]:
            self.object.translate_obj([0, 0, self.moving_speed])
            self.object1.translate_obj([0, 0, self.moving_speed])
        if key[pg.K_q]:
            self.object.translate_obj([0, self.moving_speed, 0])
            self.object1.translate_obj([0, self.moving_speed, 0])
        if key[pg.K_e]:
            self.object.translate_obj([0, -self.moving_speed, 0])
            self.object1.translate_obj([0, -self.moving_speed, 0])

        if key[pg.K_LEFT]:
            self.object.rotate_y_obj(self.rotating_speed)
            self.object1.rotate_y_obj(self.rotating_speed)
            
        if key[pg.K_RIGHT]:
            self.object.rotate_y_obj(-self.rotating_speed)
            self.object1.rotate_y_obj(-self.rotating_speed)
            
        if key[pg.K_UP]:
            self.object.rotate_x_obj(self.rotating_speed)
            self.object1.rotate_x_obj(self.rotating_speed)
            
        if key[pg.K_DOWN]:
            self.object.rotate_x_obj(-self.rotating_speed)
            self.object1.rotate_x_obj(-self.rotating_speed)
            
        if key[pg.K_8]:
            self.object.rotate_z_obj(self.rotating_speed)
            self.object1.rotate_z_obj(self.rotating_speed)
            
        if key[pg.K_9]:
            self.object.rotate_z_obj(-self.rotating_speed)
            self.object1.rotate_z_obj(-self.rotating_speed)

        if key[pg.K_h]:
            self.camera.position += self.camera.up * self.camera.moving_speed

        if key[pg.K_g]:
            self.camera.position -= self.camera.up * self.camera.moving_speed

    def draw(self):
        self.screen.fill(pg.Color('darkslategray'))
        if (self.object1.h_v_rad_to_center[2] == 1.0 or \
            self.object1.h_v_rad_to_center[3] == 1.0) and \
            math.fabs(self.object1.h_v_rad_to_center[0]) <= self.camera.h_fov / 2 and \
            math.fabs(self.object1.h_v_rad_to_center[1]) <= self.camera.v_fov / 2:
            self.object1.draw(self.camera.position)
        
        if (self.object.h_v_rad_to_center[2] == 1.0 or \
            self.object.h_v_rad_to_center[3] == 1.0) and \
            np.abs(self.object.h_v_rad_to_center[0]) <= self.camera.h_fov / 2 and \
            np.abs(self.object.h_v_rad_to_center[1]) <= self.camera.v_fov / 2:
            self.object.draw(self.camera.position)


    def run(self):
        while True:
            self.draw()
            self.config_key_bindings()
            [exit() for i in  pg.event.get() if i.type == pg.QUIT]
            pg.display.set_caption(str(self.clock.get_fps()))
            pg.display.flip()
            self.clock.tick(self.FPS)
            
                
            # self.camera.position /= self.camera.position[3]
            # print('pos', self.camera.position)
            # print('forward', self.camera.forward)
            # print('up', self.camera.up)
            # print('right', self.camera.right)


if __name__ == '__main__':
    app = SoftwareRender()
    app.run()
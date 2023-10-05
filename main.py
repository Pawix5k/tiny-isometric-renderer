import numpy as np

from renderer import Renderer, Object3D

WHITE = [255, 255, 255]
YELLOW = [200, 200, 0]
GREEN = [0, 200, 0]
BLUE = [0, 0, 200]
RED = [200, 0, 0]
ORANGE = [200, 100, 0]


cube_points = np.array(
    [
        [-1, -1, -1],
        [-1, 1, -1],
        [1, 1, -1],
        [1, -1, -1],
        [-1, -1, 1],
        [-1, 1, 1],
        [1, 1, 1],
        [1, -1, 1],
    ]
)

cube_triangles = np.array(
    [
        # bottom
        [0, 2, 1],
        [0, 3, 2],
        # top
        [4, 5, 6],
        [4, 6, 7],
        # back
        [1, 2, 6],
        [1, 6, 5],
        # front
        [0, 4, 7],
        [0, 7, 3],
        # left
        [0, 1, 5],
        [0, 5, 4],
        # right
        [3, 7, 6],
        [3, 6, 2],
    ]
)

cube_colors = np.array(
    [
        WHITE,
        WHITE,
        WHITE,
        WHITE,
        WHITE,
        WHITE,
        WHITE,
        WHITE,
        WHITE,
        WHITE,
        WHITE,
        WHITE,
    ]
)

cube = Object3D(cube_points, cube_triangles, cube_colors)


renderer = Renderer(
    [cube],
    (-2, -2, 2, 2),
    (300, 300),
    camera_angle=np.deg2rad(45),
    camera_pitch=np.deg2rad(60),
)
renderer.render_scene("scene.png")

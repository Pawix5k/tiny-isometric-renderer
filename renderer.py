import numpy as np
from PIL import Image


class Object3D:
    def __init__(self, points, triangles, colors):
        self.points = points
        self.triangles = triangles
        self.colors = colors


class Renderer:
    def __init__(
        self,
        objects,
        viewport,
        resolution,
        camera_angle=0.0,
        camera_pitch=0.0,
    ):
        self._objects = objects
        self._viewport = viewport
        self._resolution = resolution
        self._camera_angle = camera_angle
        self._camera_pitch = camera_pitch

        resolution_x, resolution_y = self._resolution
        self._screen = np.ones((resolution_y, resolution_x, 3), "uint8") * 120
        self._z_buffer = np.ones((resolution_y, resolution_y)) * -np.inf

        x_min, y_min, x_max, y_max = self._viewport
        self._range_x = np.linspace(x_min, x_max, resolution_x)
        self._range_y = np.linspace(y_max, y_min, resolution_y)

        self._camera_dir = np.array([0, 0, -1])

    def render_scene(self, output_path):
        for object_3d in self._objects:
            self._render_object(object_3d)
        im = Image.fromarray(self._screen)
        im.save(output_path)

    def _render_object(self, object_3d):
        projected_points = self._get_object_projected_points(object_3d)
        projected_triangles = projected_points[object_3d.triangles]

        visible_mask = self._get_screen_dot_products(projected_triangles) < 0
        if not any(visible_mask):
            return

        for triangle_points, color in zip(
            projected_triangles[visible_mask], object_3d.colors[visible_mask]
        ):
            self._render_triangle(triangle_points, color)

    def _get_screen_dot_products(self, triangles):
        normals = np.cross(
            (triangles[:, 0] - triangles[:, 1]),
            (triangles[:, 2] - triangles[:, 1]),
        )
        return (self._camera_dir * normals).sum(axis=1)

    def _get_object_projected_points(self, object_3d):
        return (
            object_3d.points
            @ _get_z_rotation_matrix(self._camera_angle)
            @ _get_x_rotation_matrix(self._camera_pitch)
        )

    def _render_triangle(self, points, color):
        bounding_box = _get_bounding_box(points)
        try:
            delta_z = _calculate_delta_z(points)
        except np.linalg.LinAlgError:
            return

        for screen_x, scene_x in enumerate(self._range_x):
            if scene_x < bounding_box[0, 0] or scene_x > bounding_box[1, 0]:
                continue
            for screen_y, scene_y in enumerate(self._range_y):
                if scene_y < bounding_box[0, 1] or scene_y > bounding_box[1, 1]:
                    continue
                if not _point_in_triangle(np.array([scene_x, scene_y, 0]), points):
                    continue
                depth = (
                    points[0][2]
                    + (np.array([scene_x, scene_y]) - points[0][:2]) @ delta_z
                )
                if depth <= self._z_buffer[screen_y, screen_x]:
                    continue
                self._screen[screen_y, screen_x, :] = color
                self._z_buffer[screen_y, screen_x] = depth


def _sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - (p3[1])) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def _point_in_triangle(p, triangle):
    a, b, c = triangle
    d1 = _sign(p, a, b)
    d2 = _sign(p, b, c)
    d3 = _sign(p, c, a)

    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

    return not (has_neg and has_pos)


def _get_bounding_box(points):
    return np.array(
        [
            [np.min(points[:, 0]), np.min(points[:, 1])],
            [np.max(points[:, 0]), np.max(points[:, 1])],
        ]
    )


def _get_x_rotation_matrix(angle):
    return np.array(
        [
            [1, 0, 0],
            [0, np.cos(angle), -np.sin(angle)],
            [0, np.sin(angle), np.cos(angle)],
        ]
    )


def _get_z_rotation_matrix(angle):
    return np.array(
        [
            [np.cos(angle), -np.sin(angle), 0],
            [np.sin(angle), np.cos(angle), 0],
            [0, 0, 1],
        ]
    )


def _calculate_delta_z(points):
    v_ab = points[1] - points[0]
    v_ac = points[2] - points[0]
    slope = np.array([v_ab[:2], v_ac[:2]])
    zs = np.array([v_ab[2], v_ac[2]])
    return np.linalg.solve(slope, zs)

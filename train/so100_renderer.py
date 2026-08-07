"""Small dependency-free SO100 kinematics and software renderer.

This module deliberately does not guess dataset calibration.  It consumes URDF
joint positions in radians.  Dataset action conversion lives in
``tools/render_so100_prompt.py`` so that camera/zero fitting can replace the
initial heuristic without changing forward kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

import cv2
import numpy as np


JOINT_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


def _numbers(value: str | None, default: tuple[float, ...]) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(item) for item in value.split()], dtype=np.float64)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def _transform(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = _rpy_matrix(rpy)
    result[:3, 3] = xyz
    return result


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    one = 1.0 - c
    rotation = np.array(
        [
            [c + x * x * one, x * y * one - z * s, x * z * one + y * s],
            [y * x * one + z * s, c + y * y * one, y * z * one - x * s],
            [z * x * one - y * s, z * y * one + x * s, c + z * z * one],
        ],
        dtype=np.float64,
    )
    result = np.eye(4, dtype=np.float64)
    result[:3, :3] = rotation
    return result


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


@dataclass(frozen=True)
class Visual:
    link: str
    mesh_path: Path
    origin: np.ndarray
    scale: np.ndarray
    rgba: np.ndarray


@dataclass(frozen=True)
class Camera:
    eye: np.ndarray
    target: np.ndarray
    fov_degrees: float = 52.0


def _material_map(root: ET.Element) -> dict[str, np.ndarray]:
    materials: dict[str, np.ndarray] = {}
    for material in root.findall("material"):
        color = material.find("color")
        if color is not None and material.get("name"):
            materials[material.get("name", "")] = _numbers(
                color.get("rgba"), (0.8, 0.8, 0.8, 1.0)
            )
    return materials


class SO100Model:
    def __init__(self, urdf_path: str | Path):
        self.urdf_path = Path(urdf_path).resolve()
        root = ET.parse(self.urdf_path).getroot()
        materials = _material_map(root)
        self.links = tuple(link.get("name", "") for link in root.findall("link"))
        self.visuals: list[Visual] = []
        for link in root.findall("link"):
            link_name = link.get("name", "")
            for visual in link.findall("visual"):
                mesh = visual.find("geometry/mesh")
                if mesh is None or not mesh.get("filename"):
                    continue
                origin = visual.find("origin")
                xyz = _numbers(origin.get("xyz") if origin is not None else None, (0, 0, 0))
                rpy = _numbers(origin.get("rpy") if origin is not None else None, (0, 0, 0))
                material = visual.find("material")
                rgba = np.array((0.72, 0.72, 0.72, 1.0), dtype=np.float64)
                if material is not None:
                    color = material.find("color")
                    if color is not None:
                        rgba = _numbers(color.get("rgba"), tuple(rgba))
                    elif material.get("name") in materials:
                        rgba = materials[material.get("name", "")]
                filename = mesh.get("filename", "")
                mesh_path = (self.urdf_path.parent / filename).resolve()
                self.visuals.append(
                    Visual(
                        link=link_name,
                        mesh_path=mesh_path,
                        origin=_transform(xyz, rpy),
                        scale=_numbers(mesh.get("scale"), (1, 1, 1)),
                        rgba=rgba,
                    )
                )

        self.joints: list[Joint] = []
        for joint in root.findall("joint"):
            if joint.get("type") == "fixed":
                continue
            parent = joint.find("parent")
            child = joint.find("child")
            origin = joint.find("origin")
            axis = joint.find("axis")
            limit = joint.find("limit")
            if parent is None or child is None:
                continue
            self.joints.append(
                Joint(
                    name=joint.get("name", ""),
                    parent=parent.get("link", ""),
                    child=child.get("link", ""),
                    origin=_transform(
                        _numbers(origin.get("xyz") if origin is not None else None, (0, 0, 0)),
                        _numbers(origin.get("rpy") if origin is not None else None, (0, 0, 0)),
                    ),
                    axis=_numbers(axis.get("xyz") if axis is not None else None, (1, 0, 0)),
                    lower=float(limit.get("lower", "-inf")) if limit is not None else -np.inf,
                    upper=float(limit.get("upper", "inf")) if limit is not None else np.inf,
                )
            )
        self.joint_map = {joint.name: joint for joint in self.joints}
        missing = [name for name in JOINT_NAMES if name not in self.joint_map]
        if missing:
            raise ValueError(f"URDF is missing SO100 joints: {missing}")

        children = {joint.child for joint in self.joints}
        roots = [name for name in self.links if name not in children]
        if not roots:
            raise ValueError("Could not find URDF root link")
        self.root_link = roots[0]

    def forward_kinematics(self, q: np.ndarray) -> dict[str, np.ndarray]:
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (len(JOINT_NAMES),):
            raise ValueError(f"q must have shape (6,), got {q.shape}")
        q_by_name = dict(zip(JOINT_NAMES, q.tolist()))
        transforms = {self.root_link: np.eye(4, dtype=np.float64)}
        remaining = list(self.joints)
        while remaining:
            progressed = False
            for joint in remaining[:]:
                if joint.parent not in transforms:
                    continue
                angle = q_by_name.get(joint.name, 0.0)
                transforms[joint.child] = (
                    transforms[joint.parent]
                    @ joint.origin
                    @ _axis_rotation(joint.axis, angle)
                )
                remaining.remove(joint)
                progressed = True
            if not progressed:
                raise ValueError("URDF joint graph is disconnected or cyclic")
        return transforms

    def joint_positions(self, q: np.ndarray) -> dict[str, np.ndarray]:
        transforms = self.forward_kinematics(q)
        result = {self.root_link: transforms[self.root_link][:3, 3].copy()}
        for joint in self.joints:
            result[joint.name] = transforms[joint.child][:3, 3].copy()
        return result

    def render(
        self,
        q: np.ndarray,
        camera: Camera,
        width: int = 512,
        height: int = 320,
        background: tuple[int, int, int] = (0, 0, 0),
        skeleton: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        transforms = self.forward_kinematics(q)
        image = np.empty((height, width, 3), dtype=np.uint8)
        image[:] = np.asarray(background, dtype=np.uint8)
        depth = np.full((height, width), np.inf, dtype=np.float32)
        view = _view_matrix(camera)
        focal = 0.5 * width / np.tan(np.deg2rad(camera.fov_degrees) * 0.5)

        triangles: list[tuple[float, np.ndarray, tuple[int, int, int]]] = []
        for visual in self.visuals:
            vertices, faces = _load_mesh(visual.mesh_path)
            vertices = vertices * visual.scale[None]
            world = transforms[visual.link] @ visual.origin
            homogeneous = np.concatenate([vertices, np.ones((len(vertices), 1))], axis=1)
            world_vertices = (world @ homogeneous.T).T[:, :3]
            camera_vertices = (view[:3, :3] @ world_vertices.T).T + view[:3, 3]
            valid = camera_vertices[:, 2] > 1e-4
            projected = np.empty((len(vertices), 2), dtype=np.float64)
            projected[:, 0] = focal * camera_vertices[:, 0] / np.maximum(camera_vertices[:, 2], 1e-4) + width / 2
            projected[:, 1] = focal * camera_vertices[:, 1] / np.maximum(camera_vertices[:, 2], 1e-4) + height / 2
            color = tuple(int(np.clip(channel * 255, 0, 255)) for channel in visual.rgba[:3][::-1])
            for face in faces:
                if not valid[face].all():
                    continue
                points = projected[face]
                if (
                    points[:, 0].max() < 0
                    or points[:, 0].min() >= width
                    or points[:, 1].max() < 0
                    or points[:, 1].min() >= height
                ):
                    continue
                triangles.append((float(camera_vertices[face, 2].mean()), points, color))
        # A depth-sort is sufficient for the diagnostic control render and keeps
        # this module independent of OpenGL/EGL availability on the cluster.
        for mean_depth, points, color in sorted(triangles, key=lambda item: item[0], reverse=True):
            polygon = np.round(points).astype(np.int32)
            cv2.fillConvexPoly(image, polygon, color, lineType=cv2.LINE_AA)
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillConvexPoly(mask, polygon, 1)
            depth[mask.astype(bool)] = np.minimum(depth[mask.astype(bool)], mean_depth)

        if skeleton:
            positions = self.joint_positions(q)
            ordered = [positions[self.root_link]] + [positions[name] for name in JOINT_NAMES]
            points = _project_points(np.asarray(ordered), view, focal, width, height)
            for index in range(len(points) - 1):
                cv2.line(image, tuple(points[index]), tuple(points[index + 1]), (0, 255, 255), 3, cv2.LINE_AA)
            for point in points:
                cv2.circle(image, tuple(point), 4, (0, 0, 255), -1, cv2.LINE_AA)
        return image, depth

    def render_orthographic(
        self,
        q: np.ndarray,
        rotation: np.ndarray,
        pixels_per_meter: float,
        translation: np.ndarray,
        width: int = 256,
        height: int = 160,
        background: tuple[int, int, int] = (0, 0, 0),
    ) -> tuple[np.ndarray, np.ndarray]:
        """Render with the weak-perspective camera used by the alignment gate."""
        transforms = self.forward_kinematics(q)
        rotation = np.asarray(rotation, dtype=np.float64)
        translation = np.asarray(translation, dtype=np.float64)
        image = np.empty((height, width, 3), dtype=np.uint8)
        image[:] = np.asarray(background, dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        triangles: list[tuple[float, np.ndarray, tuple[int, int, int]]] = []
        for visual in self.visuals:
            vertices, faces = _load_mesh(visual.mesh_path)
            vertices = vertices * visual.scale[None]
            world = transforms[visual.link] @ visual.origin
            homogeneous = np.concatenate([vertices, np.ones((len(vertices), 1))], axis=1)
            world_vertices = (world @ homogeneous.T).T[:, :3]
            camera_vertices = (rotation @ world_vertices.T).T
            projected = camera_vertices[:, :2] * pixels_per_meter + translation
            color = tuple(
                int(np.clip(channel * 255, 0, 255)) for channel in visual.rgba[:3][::-1]
            )
            for face in faces:
                points = projected[face]
                if (
                    points[:, 0].max() < 0
                    or points[:, 0].min() >= width
                    or points[:, 1].max() < 0
                    or points[:, 1].min() >= height
                ):
                    continue
                triangles.append((float(camera_vertices[face, 2].mean()), points, color))
        for mean_depth, points, color in sorted(triangles, key=lambda item: item[0]):
            polygon = np.round(points).astype(np.int32)
            cv2.fillConvexPoly(image, polygon, color, lineType=cv2.LINE_AA)
            cv2.fillConvexPoly(mask, polygon, 255, lineType=cv2.LINE_AA)
        return image, mask

    def project_orthographic_joints(
        self,
        q: np.ndarray,
        rotation: np.ndarray,
        pixels_per_meter: float,
        translation: np.ndarray,
    ) -> np.ndarray:
        """Project the SO-100 kinematic chain with a weak-perspective camera.

        The returned order is ``root, shoulder_pan, ..., gripper``.  Keeping
        this operation separate from rasterization lets all calibration
        methods optimize exactly the geometry later consumed by OSCAR.
        """
        positions = self.joint_positions(q)
        points = np.asarray(
            [positions[self.root_link]] + [positions[name] for name in JOINT_NAMES],
            dtype=np.float64,
        )
        camera = (np.asarray(rotation, dtype=np.float64) @ points.T).T
        return camera[:, :2] * float(pixels_per_meter) + np.asarray(
            translation, dtype=np.float64
        )

    def render_oscar_skeleton_orthographic(
        self,
        q: np.ndarray,
        rotation: np.ndarray,
        pixels_per_meter: float,
        translation: np.ndarray,
        width: int = 256,
        height: int = 160,
    ) -> np.ndarray:
        """Render an OSCAR-compatible RGB skeleton on black.

        OSCAR's public renderer uses yellow arm links, blue joint dots, red
        gripper fingers, and RGB end-effector axes.  This function returns an
        *RGB* array; callers should not reverse its channels before writing it
        with imageio/PyAV.
        """
        rotation = np.asarray(rotation, dtype=np.float64)
        translation = np.asarray(translation, dtype=np.float64)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        joints = self.project_orthographic_joints(
            q, rotation, pixels_per_meter, translation
        )

        def pixel(point: np.ndarray) -> tuple[int, int]:
            return tuple(np.round(point).astype(np.int32))

        for start, end in zip(joints[:-1], joints[1:]):
            cv2.line(frame, pixel(start), pixel(end), (255, 255, 0), 3, cv2.LINE_AA)
        for point in joints:
            cv2.circle(frame, pixel(point), 4, (0, 0, 255), -1, cv2.LINE_AA)

        transforms = self.forward_kinematics(q)
        fixed_jaw = transforms.get("Fixed_Jaw")
        moving_jaw = transforms.get("Moving_Jaw")
        if fixed_jaw is not None:
            axis_length = 0.045
            local_axes = np.asarray(
                [
                    (0.0, 0.0, 0.0),
                    (axis_length, 0.0, 0.0),
                    (0.0, axis_length, 0.0),
                    (0.0, 0.0, axis_length),
                    (0.01, -0.097, 0.0),
                ],
                dtype=np.float64,
            )
            fixed_world = (
                fixed_jaw[:3, :3] @ local_axes.T + fixed_jaw[:3, 3:4]
            ).T
            extra_world = list(fixed_world)
            if moving_jaw is not None:
                moving_tip = moving_jaw[:3, :3] @ np.asarray(
                    (-0.01, -0.073, 0.0), dtype=np.float64
                ) + moving_jaw[:3, 3]
                extra_world.append(moving_tip)
            camera = (rotation @ np.asarray(extra_world).T).T
            uv = camera[:, :2] * float(pixels_per_meter) + translation
            origin = pixel(uv[0])
            # cv2 writes raw tuples into this RGB canvas, matching OSCAR's
            # public imageio renderer rather than OpenCV's usual BGR convention.
            for endpoint, color in zip(
                uv[1:4], ((0, 0, 255), (0, 255, 0), (255, 0, 0))
            ):
                cv2.arrowedLine(
                    frame, origin, pixel(endpoint), color, 4, cv2.LINE_AA, tipLength=0.15
                )
            cv2.arrowedLine(
                frame, origin, pixel(uv[4]), (255, 0, 0), 5, cv2.LINE_AA, tipLength=0.15
            )
            if len(uv) > 5:
                cv2.arrowedLine(
                    frame,
                    origin,
                    pixel(uv[5]),
                    (255, 0, 0),
                    5,
                    cv2.LINE_AA,
                    tipLength=0.15,
                )
        return frame


def _view_matrix(camera: Camera) -> np.ndarray:
    eye = np.asarray(camera.eye, dtype=np.float64)
    target = np.asarray(camera.target, dtype=np.float64)
    forward = target - eye
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    world_up = np.array((0.0, 0.0, 1.0))
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-6:
        world_up = np.array((0.0, 1.0, 0.0))
        right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    # Image y increases downward.
    rotation = np.stack([right, -up, forward], axis=0)
    view = np.eye(4, dtype=np.float64)
    view[:3, :3] = rotation
    view[:3, 3] = -rotation @ eye
    return view


def _project_points(
    points: np.ndarray, view: np.ndarray, focal: float, width: int, height: int
) -> np.ndarray:
    camera = (view[:3, :3] @ points.T).T + view[:3, 3]
    z = np.maximum(camera[:, 2], 1e-4)
    result = np.stack(
        [focal * camera[:, 0] / z + width / 2, focal * camera[:, 1] / z + height / 2],
        axis=1,
    )
    return np.round(result).astype(np.int32)


@lru_cache(maxsize=64)
def _load_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    suffix = path.suffix.lower()
    if suffix == ".stl":
        return _load_binary_stl(path)
    if suffix == ".ply":
        return _load_ascii_ply(path)
    raise ValueError(f"Unsupported mesh format: {path}")


def _load_binary_stl(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = path.read_bytes()
    if len(data) < 84:
        raise ValueError(f"Invalid STL: {path}")
    count = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + count * 50
    if expected != len(data):
        raise ValueError(f"Only binary STL is supported: {path}")
    vertices = np.empty((count * 3, 3), dtype=np.float64)
    for index in range(count):
        offset = 84 + index * 50 + 12
        vertices[index * 3 : index * 3 + 3] = np.frombuffer(
            data, dtype="<f4", count=9, offset=offset
        ).reshape(3, 3)
    faces = np.arange(count * 3, dtype=np.int32).reshape(count, 3)
    return vertices, faces


def _load_ascii_ply(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text().splitlines()
    vertex_count = face_count = 0
    header_end = None
    for index, line in enumerate(lines):
        if line.startswith("element vertex "):
            vertex_count = int(line.split()[-1])
        elif line.startswith("element face "):
            face_count = int(line.split()[-1])
        elif line == "end_header":
            header_end = index + 1
            break
    if header_end is None:
        raise ValueError(f"Invalid PLY: {path}")
    vertices = np.asarray(
        [[float(value) for value in line.split()[:3]] for line in lines[header_end : header_end + vertex_count]],
        dtype=np.float64,
    )
    faces = []
    for line in lines[header_end + vertex_count : header_end + vertex_count + face_count]:
        values = [int(value) for value in line.split()]
        if values[0] == 3:
            faces.append(values[1:4])
        elif values[0] > 3:
            for index in range(1, values[0] - 1):
                faces.append([values[1], values[index + 1], values[index + 2]])
    return vertices, np.asarray(faces, dtype=np.int32)

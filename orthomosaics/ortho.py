from cv2 import perspectiveTransform, warpPerspective
from numpy import array, cos, linalg, matrix, ndarray, pi, radians, sin, sqrt

from orthomosaics.utils.schemas import Camera, ImageMetadata, OrthorectificationMetadata


def focal_length_pixels(image_width: int, camera_settings: Camera) -> int:
    return (
        image_width
        * camera_settings.focal_length_mm
        / (camera_settings.ccd_width_pixels * camera_settings.pixel_width_mm)
    )


def camera_intrinsics_matrix(
    image_width: int, image_height: int, camera_settings: Camera
) -> ndarray:
    focal_length_x = focal_length_y = focal_length_pixels(
        image_width=image_width,
        camera_settings=camera_settings,
    )
    return array(
        (
            (focal_length_x, 0, image_width / 2),
            (0, focal_length_y, image_height / 2),
            (0, 0, 1),
        )
    )


def rotation_matrix(roll: radians, pitch: radians) -> ndarray:
    rotation_matrix_x_axis = array(
        (
            (1, 0, 0),
            (0, cos(pitch), -sin(pitch)),
            (0, sin(pitch), cos(pitch)),
        )
    )
    rotation_matrix_y_axis = array(
        ((cos(roll), 0, -sin(roll)), (0, 1, 0), (sin(roll), 0, cos(roll)))
    )
    rotation_matrix_z_axis = array(((1, 0, 0), (0, 1, 0), (0, 0, 1)))
    return rotation_matrix_x_axis @ rotation_matrix_y_axis @ rotation_matrix_z_axis


def translation_matrix(
    pitch: radians, distance_camera_ground_mm: int, camera_settings: Camera
) -> ndarray:
    distance_horizontal_between_real_and_virtual_camera_mm = sqrt(
        distance_camera_ground_mm**2 - camera_settings.camera_height_mm**2
    )
    distance_vertical_between_real_and_virtual_camera_mm = 0
    translate = matrix(
        (
            0,
            distance_horizontal_between_real_and_virtual_camera_mm,
            distance_vertical_between_real_and_virtual_camera_mm,
        )
    )
    normal = matrix((0, cos(pitch), sin(pitch)))
    return translate.T * normal / camera_settings.camera_height_mm


def transformation_matrix(
    image_height: int,
    image_width: int,
    image_metadata: ImageMetadata,
    camera_settings: Camera,
) -> ndarray:
    pitch = radians(image_metadata.pitch_deg)
    distance_camera_ground_mm = camera_settings.camera_height_mm / cos(-pitch)
    T = translation_matrix(
        pitch=pitch,
        distance_camera_ground_mm=distance_camera_ground_mm,
        camera_settings=camera_settings,
    )
    R = rotation_matrix(
        roll=radians(image_metadata.roll_deg),
        pitch=radians(-90 - image_metadata.pitch_deg),
    )
    C = camera_intrinsics_matrix(
        image_width=image_width,
        image_height=image_height,
        camera_settings=camera_settings,
    )
    return C @ (R + T) @ linalg.inv(C)


def corner_points(image_width: int, image_height: int) -> ndarray:
    return array(
        ([[[0, 0], [image_width, 0], [0, image_height], [image_width, image_height]]]),
        dtype="float32",
    )


def second_translation_matrix(
    transformation_matrix: ndarray, image_width: int, image_height: int
) -> tuple[ndarray, tuple[int, int]]:
    trans_points = perspectiveTransform(
        corner_points(image_width=image_width, image_height=image_height),
        transformation_matrix,
    )[0]
    new_left = min(trans_points[0, 0], trans_points[2, 0])
    new_top = min(trans_points[0, 1], trans_points[1, 1])
    dx, dy = -new_left, -new_top
    translate = array(((1, 0, dx), (0, 1, dy), (0, 0, 1)))
    new_right = max(trans_points[1, 0], trans_points[3, 0])
    new_bottom = max(trans_points[2, 1], trans_points[3, 1])
    new_size = int(new_right - new_left), int(new_bottom - new_top)
    return translate, new_size


def homography_matrix(
    image_width: int,
    image_height: int,
    image_metadata: ImageMetadata,
    camera_settings: Camera,
) -> tuple[ndarray, tuple[int, int]]:
    transformation = transformation_matrix(
        image_width=image_width,
        image_height=image_height,
        image_metadata=image_metadata,
        camera_settings=camera_settings,
    )
    translation, new_size = second_translation_matrix(
        transformation_matrix=transformation,
        image_width=image_width,
        image_height=image_height,
    )
    homography = translation @ transformation
    return homography, new_size


def metres_per_pixel_y_axis(
    image_width: int,
    image_height: int,
    homography: ndarray,
    image_metadata: ImageMetadata,
    camera_settings: Camera,
) -> float:
    centre_point = [image_width / 2, image_height / 2]
    bottom_centre_point = [image_width / 2, image_height]
    left_centre_point = [0, image_height / 2]
    points = array(([[centre_point, bottom_centre_point, left_centre_point]]))
    points_projected = perspectiveTransform(points, homography)[0]
    distance_focal_centre_to_bottom_centre = linalg.norm(
        points_projected[1] - points_projected[0]
    )
    pitch = radians(image_metadata.pitch_deg)
    angle_principle_axis_ground_plane_intersect = pi / 2 + pitch
    principle_bottom_angle = radians(camera_settings.vertical_fov / 2)
    bottom_ground_angle = (
        pi - principle_bottom_angle - angle_principle_axis_ground_plane_intersect
    )
    distance_camera_ground_mm = camera_settings.camera_height_mm / cos(-pitch)
    distance_focal_centre_to_bottom_centre_mm = (
        distance_camera_ground_mm
        * sin(principle_bottom_angle)
        / sin(bottom_ground_angle)
    )
    return (
        distance_focal_centre_to_bottom_centre_mm
        / 1000
        / distance_focal_centre_to_bottom_centre
    )


def orthorectify_image(
    image: ndarray,
    image_metadata: ImageMetadata,
    camera_settings: Camera,
) -> tuple[ndarray, OrthorectificationMetadata]:
    height, width, _ = image.shape
    homography, new_size = homography_matrix(
        image_width=width,
        image_height=height,
        image_metadata=image_metadata,
        camera_settings=camera_settings,
    )
    orthorectified_image = warpPerspective(image, homography, new_size)
    orthorectification_metadata = OrthorectificationMetadata(
        metres_per_pixel=metres_per_pixel_y_axis(
            image_width=width,
            image_height=height,
            homography=homography,
            image_metadata=image_metadata,
            camera_settings=camera_settings,
        ),
    )
    return orthorectified_image, orthorectification_metadata

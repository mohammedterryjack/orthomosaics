from datetime import datetime

from cv2 import COLOR_RGB2RGBA, cvtColor
from geopandas import GeoDataFrame
from matplotlib.pyplot import show, subplots
from numpy import ndarray, ogrid, sqrt, zeros
from scipy.ndimage import rotate
from shapely.geometry import Point

from orthomosaics.ortho import orthorectify_image
from orthomosaics.utils.schemas import (
    GPS,
    Camera,
    ImageMetadata,
    OrthomosaicMetadata,
    OrthorectificationMetadata,
)


def update_roi(tile: ndarray, x: int, y: int, image: ndarray) -> None:
    height, width, _ = image.shape
    tile[y : y + height, x : x + width, :] = update_roi_by_dominant_alpha_channel(
        roi=tile[y : y + height, x : x + width, :].copy(), image=image.copy()
    )


def update_roi_by_dominant_alpha_channel(roi: ndarray, image: ndarray) -> None:
    alpha_roi = roi[:, :, 3]
    alpha_image = image[:, :, 3]
    roi_new = roi_masked(roi=image, pixel_weights=alpha_image > alpha_roi)
    roi_old = roi_masked(roi=roi, pixel_weights=alpha_image <= alpha_roi)
    return roi_old + roi_new


def roi_masked(roi: ndarray, pixel_weights: ndarray) -> ndarray:
    roi[:, :, 0] *= pixel_weights
    roi[:, :, 1] *= pixel_weights
    roi[:, :, 2] *= pixel_weights
    roi[:, :, 3] *= pixel_weights
    return roi


def convert_gps_degrees_to_metres(
    coordinates: list[Point],
) -> list[tuple[float, float]]:
    data = GeoDataFrame(coordinates, columns=["geometry"])
    return list(zip(data.geometry.x, data.geometry.y))


def convert_m_to_pixels(m: float, m_per_pixel: float) -> int:
    return int(m / m_per_pixel)


def convert_coordinates_to_pixels(
    coordinates: list[tuple[float, float]], metadata: list[OrthorectificationMetadata]
) -> list[tuple[int, int]]:
    return [
        (int(x / meta.metres_per_pixel), int(y / meta.metres_per_pixel))
        for (x, y), meta in zip(coordinates, metadata)
    ]


def intensity_gradient_from_focal_point(
    image_height: int, image_width: int, focal_point_x: int, focal_point_y: int
) -> ndarray:
    y_axis, x_axis = ogrid[:image_height, :image_width]
    inverse_matrix = sqrt((x_axis - focal_point_x) ** 2 + (y_axis - focal_point_y) ** 2)
    inverse_matrix_normalised = inverse_matrix / inverse_matrix.max()
    return 1 - inverse_matrix_normalised


def add_alpha_channel_and_rotate(
    image: ndarray,
    heading: float,
    bottom_crop: int,
    side_crop: int,
    display: bool,
) -> ndarray:
    height, width, _ = image.shape
    alpha = (
        intensity_gradient_from_focal_point(
            image_width=width,
            image_height=height,
            focal_point_x=width // 2,
            focal_point_y=height,
        )
        * 255
    )
    alpha[:, :side_crop] = 0
    alpha[:, -side_crop:] = 0
    alpha[height - bottom_crop :, :] = 255 // 2
    image_rgba = cvtColor(image, COLOR_RGB2RGBA)
    image_rgba[:, :, 3] = alpha
    image_rgba_rotated = rotate(image_rgba, -heading, reshape=True, order=0)
    image_rgba_rotated[:, :, 3] = (
        image_rgba_rotated[:, :, :3].sum(axis=-1).astype(bool)
        * image_rgba_rotated[:, :, 3]
    )
    if display:
        _, ax = subplots(nrows=1, ncols=2)
        ax[0].imshow(image)
        ax[1].imshow(image_rgba_rotated)
        show()
    return image_rgba_rotated[::-1, :, :]


def add_to_orthomosaic(
    orthomosaic_image: ndarray | None,
    orthomosaic_metadata: OrthomosaicMetadata | None,
    backdown_image: ndarray,
    gps_data: GPS,
    backdown_image_metadata: ImageMetadata,
    camera_settings: Camera,
    bottom_crop: int,
    side_crop: int,
) -> tuple[ndarray, OrthomosaicMetadata]:
    """Add another backdown image to the orthomosaic"""
    orthorectified_image, orthorectification_metadata = orthorectify_image(
        image=backdown_image,
        image_metadata=backdown_image_metadata,
        camera_settings=camera_settings,
    )
    rotated_orthorectified_image = add_alpha_channel_and_rotate(
        image=orthorectified_image,
        heading=gps_data.heading,
        bottom_crop=bottom_crop,
        side_crop=side_crop,
        display=False,
    )
    x_m, y_m = convert_gps_degrees_to_metres(
        coordinates=[Point((gps_data.x, gps_data.y))]
    )[0]
    rotated_orthorectified_image_metadata = OrthomosaicMetadata(
        x_m=x_m,
        y_m=y_m,
        x_m_per_pixel=orthorectification_metadata.metres_per_pixel,
        y_m_per_pixel=orthorectification_metadata.metres_per_pixel,
        id=hash(datetime.now()),
    )
    if orthomosaic_image is None:
        return rotated_orthorectified_image, rotated_orthorectified_image_metadata

    assert orthomosaic_metadata is not None

    updated_x_min = min(
        orthomosaic_metadata.x_m,
        rotated_orthorectified_image_metadata.x_m,
    )
    updated_y_min = min(
        orthomosaic_metadata.y_m,
        rotated_orthorectified_image_metadata.y_m,
    )
    updated_orthomosaic_metadata = OrthomosaicMetadata(
        x_m=updated_x_min,
        y_m=updated_y_min,
        x_m_per_pixel=orthomosaic_metadata.x_m_per_pixel
        if updated_x_min == orthomosaic_metadata.x_m
        else rotated_orthorectified_image_metadata.x_m_per_pixel,
        y_m_per_pixel=orthomosaic_metadata.y_m_per_pixel
        if updated_y_min == orthomosaic_metadata.y_m
        else rotated_orthorectified_image_metadata.y_m_per_pixel,
        id=orthomosaic_metadata.id,
    )

    # Get relative position (in pixels) of new image to add

    rotated_x_relative = (
        rotated_orthorectified_image_metadata.x_m - updated_orthomosaic_metadata.x_m
    )
    rotated_y_relative = (
        rotated_orthorectified_image_metadata.y_m - updated_orthomosaic_metadata.y_m
    )

    rotated_height_pixels, rotated_width_pixels, _ = rotated_orthorectified_image.shape
    rotated_x_min_pixels = convert_m_to_pixels(
        m=rotated_x_relative,
        m_per_pixel=rotated_orthorectified_image_metadata.x_m_per_pixel,
    )
    rotated_y_min_pixels = convert_m_to_pixels(
        m=rotated_y_relative,
        m_per_pixel=rotated_orthorectified_image_metadata.y_m_per_pixel,
    )
    rotated_x_max_pixels = rotated_x_min_pixels + rotated_width_pixels
    rotated_y_max_pixels = rotated_y_min_pixels + rotated_height_pixels

    # Get relative position (in pixels) of existing mosaic to readd

    ortho_x_relative = orthomosaic_metadata.x_m - updated_orthomosaic_metadata.x_m
    ortho_y_relative = orthomosaic_metadata.y_m - updated_orthomosaic_metadata.y_m

    ortho_height_pixels, ortho_width_pixels, _ = orthomosaic_image.shape
    ortho_x_min_pixels = convert_m_to_pixels(
        m=ortho_x_relative, m_per_pixel=orthomosaic_metadata.x_m_per_pixel
    )
    ortho_y_min_pixels = convert_m_to_pixels(
        m=ortho_y_relative, m_per_pixel=orthomosaic_metadata.y_m_per_pixel
    )
    ortho_x_max_pixels = ortho_x_min_pixels + ortho_width_pixels
    ortho_y_max_pixels = ortho_y_min_pixels + ortho_height_pixels

    updated_x_max_pixels = max(ortho_x_max_pixels, rotated_x_max_pixels)
    updated_y_max_pixels = max(ortho_y_max_pixels, rotated_y_max_pixels)

    # Get new size of mosaic

    updated_x_min_pixels = 0
    updated_y_min_pixels = 0
    updated_width = updated_x_max_pixels - updated_x_min_pixels
    updated_height = updated_y_max_pixels - updated_y_min_pixels

    updated_orthomosaic_image = zeros(
        shape=(updated_height, updated_width, 4), dtype="uint8"
    )

    update_roi(
        tile=updated_orthomosaic_image,
        image=orthomosaic_image,
        x=ortho_x_min_pixels,
        y=ortho_y_min_pixels,
    )
    update_roi(
        tile=updated_orthomosaic_image,
        image=rotated_orthorectified_image,
        x=rotated_x_min_pixels,
        y=rotated_y_min_pixels,
    )

    return updated_orthomosaic_image, updated_orthomosaic_metadata

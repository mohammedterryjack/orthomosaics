from base64 import b64decode, b64encode
from io import BytesIO

from cv2 import imread
from numpy import array, ndarray
from PIL import Image
from pydantic import BaseModel

from orthomosaics.utils.schemas import GPS, Camera, ImageMetadata, OrthomosaicMetadata

Image.MAX_IMAGE_PIXELS = None


mx9_camera = Camera(
    focal_length_mm=-8.27497,
    ccd_width_pixels=4096,
    pixel_width_mm=0.00345,
    camera_height_mm=2048,
    vertical_fov=55,
)


class NullImage(Exception):
    pass


class Results(BaseModel):
    message: str
    orthomosaic_metadata: OrthomosaicMetadata | None


class Payload(BaseModel):
    backdown_image_b64: str
    backdown_image_metadata: ImageMetadata
    gps: GPS
    camera_settings: Camera = mx9_camera
    side_crop_pixels: int = 1000
    bottom_crop_pixels: int = 0
    orthomosaic_metadata: OrthomosaicMetadata | None = None


def read_image(image_path: str) -> ndarray:
    image = imread(image_path)
    if image is None:
        raise NullImage(f"The image {image_path} does not exist")
    return image


def array_to_bytes(image: ndarray) -> bytes:
    image_temp = Image.fromarray(image)
    buffer = BytesIO()
    image_temp.save(buffer, format="PNG")
    return buffer.getvalue()


def bytes_to_array(image_bytes: BytesIO) -> ndarray:
    return array(Image.open(image_bytes))


def encode_image(image_path: str) -> str:
    image_arr = read_image(image_path=image_path)
    image_bytes = array_to_bytes(image=image_arr)
    return b64encode(image_bytes).decode()


def decode_image(image_b64: str) -> ndarray:
    image_bytes = BytesIO(b64decode(image_b64))
    return bytes_to_array(image_bytes=image_bytes)

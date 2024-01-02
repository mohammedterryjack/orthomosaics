from pydantic import BaseModel


class GPS(BaseModel):
    x: float
    y: float
    heading: float


class Camera(BaseModel):
    focal_length_mm: float
    ccd_width_pixels: int
    pixel_width_mm: float
    camera_height_mm: int
    vertical_fov: int


class ImageMetadata(BaseModel):
    roll_deg: float
    pitch_deg: float


class OrthorectificationMetadata(BaseModel):
    metres_per_pixel: float


class OrthomosaicMetadata(BaseModel):
    id: int
    x_m: float
    y_m: float
    x_m_per_pixel: float
    y_m_per_pixel: float

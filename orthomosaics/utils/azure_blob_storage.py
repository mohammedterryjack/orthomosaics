from io import BytesIO

from azure.core.exceptions import ResourceNotFoundError
from azure.storage.blob import BlobServiceClient
from numpy import ndarray

from orthomosaics.utils.rest_api import array_to_bytes, bytes_to_array
from orthomosaics.utils.schemas import OrthomosaicMetadata


class AzureStorage:
    def __init__(self, connection_string: str, container_name: str) -> None:
        client = BlobServiceClient.from_connection_string(connection_string)
        self.container = client.get_container_client(container_name)
        if not self.container.exists():
            raise ResourceNotFoundError

    def download_image(self, metadata: OrthomosaicMetadata) -> ndarray:
        image_bytes = self._image_bytes(image_id=metadata.id)
        return bytes_to_array(image_bytes=image_bytes)

    def upload_image(self, image: ndarray, metadata: OrthomosaicMetadata) -> None:
        self.container.upload_blob(
            name=f"orthomosaic_{metadata.id}.png",
            data=array_to_bytes(image=image),
            overwrite=True,
        )

    def _image_bytes(self, image_id: int) -> BytesIO:
        image_bytes = BytesIO()
        self.container.download_blob(f"orthomosaic_{image_id}.png").readinto(
            image_bytes
        )
        return image_bytes

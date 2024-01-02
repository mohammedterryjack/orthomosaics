from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from starlette.status import HTTP_302_FOUND
from uvicorn import run

from orthomosaics.mosaics import add_to_orthomosaic
from orthomosaics.utils.azure_blob_storage import AzureStorage
from orthomosaics.utils.rest_api import Payload, Results, decode_image

storage = AzureStorage(
    container_name="YOUR_CONTAINER_NAME",
    connection_string="YOUR_AZURE_CONNECTION_STRING",
)
app = FastAPI(title="Orthomosaics", debug=False, version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def main() -> RedirectResponse:
    return RedirectResponse(url="/docs", status_code=HTTP_302_FOUND)


@app.get(path="/orthomosaic/", description="download an orthomosaic image")
async def download_orthmosaic_image(orthomosaic_id: int) -> bytes:
    try:
        return storage._image_bytes(image_id=orthomosaic_id).get_value()
    except Exception as error_details:
        raise HTTPException(status_code=500, detail=str(error_details))


@app.post(path="/orthomosaic/", description="add a backdown image to the orthomosaic")
async def upload_backdown_image(payload: Payload) -> Results:
    try:
        new_orthomosaic_image, new_orthomosaic_metadata = add_to_orthomosaic(
            orthomosaic_image=storage.download_image(
                metadata=payload.orthomosaic_metadata
            )
            if payload.orthomosaic_metadata
            else None,
            orthomosaic_metadata=payload.orthomosaic_metadata,
            backdown_image=decode_image(image_b64=payload.backdown_image_b64),
            gps_data=payload.gps,
            backdown_image_metadata=payload.backdown_image_metadata,
            camera_settings=payload.camera_settings,
            bottom_crop=payload.bottom_crop_pixels,
            side_crop=payload.side_crop_pixels,
        )
        storage.upload_image(
            image=new_orthomosaic_image, metadata=new_orthomosaic_metadata
        )
        return Results(
            message=f"Updated orthomosaic uploaded to Azure Storage Blob: {storage.container}/orthomosaic_{new_orthomosaic_metadata.id}.png",
            orthomosaic_metadata=new_orthomosaic_metadata,
        )
    except Exception as error_details:
        raise HTTPException(status_code=500, detail=str(error_details))


if __name__ == "__main__":
    run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )

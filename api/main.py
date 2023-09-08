from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder

import uvicorn

from services.inference import read_file_as_image, Model
from services.pipeline import prepare_image


app = FastAPI()


@app.get("/health-check")
async def health_check():
    return {"message": "alive"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
  try:
    file_content = await file.read()
    image = read_file_as_image(file_content)
    label,confidence = Model.predict(prepare_image(image))
    response_data = {"status":"ok","label": label,"confidence":confidence,"mesage":"prediction successful"}
    return JSONResponse(content=jsonable_encoder(response_data))
  
  except Exception as e:
    response_data = {"status":"error","label": "","confidence":None,"message": f"{e}"}
    return JSONResponse(content=jsonable_encoder(response_data),status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)

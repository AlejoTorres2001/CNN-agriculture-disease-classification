from pydantic import BaseSettings


class Settings(BaseSettings):
    CLASS_NAMES: list[str] = ['Potato___Early_blight',
                              'Potato___Late_blight', 'Potato___healthy']
    IMG_SIZE: int = 226
    CHANNELS: int = 3
    MODEL_PATH: str = "../models/1"


settings = Settings()

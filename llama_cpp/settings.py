from pydantic import BaseModel, Field
from dotenv import load_dotenv


class AppConfig(BaseModel):
    
    API_URL: str = Field("http://127.0.0.1:8000", env="API_URL")
    API_TOKEN: str = Field("very-secret-token", env="API_TOKEN")
    MODEL_PATH: str = Field("Q8_0.gguf", env="MODEL_PATH")
    FILE_PATH: str = Field("chats-aia/chats-aia/chats/", env="FILE_PATH")
    OUT_FILE: str = Field("results.json", env="OUT_FILE")

    @classmethod
    def from_env(cls):
        load_dotenv()
        return cls.model_validate({})

if __name__ == "__main__":
    config = AppConfig.from_env()
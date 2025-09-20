from pydantic import BaseModel, Field
from dotenv import load_dotenv


class AppConfig(BaseModel):
    
    API_URL: str = Field("http://127.0.0.1:8000", env="API_URL")
    API_TOKEN: str = Field("very-secret-token", env="API_TOKEN")

    @classmethod
    def from_env(cls):
        load_dotenv()
        return cls.model_validate({})

if __name__ == "__main__":
    config = AppConfig.from_env()
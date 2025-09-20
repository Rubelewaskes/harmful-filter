from pydantic import BaseModel, Field
from dotenv import load_dotenv


class AppConfig(BaseModel):
    
    API_URL: str = Field("http://195.209.210.246:8000", env="API_URL")
    API_TOKEN: str = Field("very-secret-token", env="API_TOKEN")
    MODEL_PATH: str = Field("Q4_K_M.gguf", env="MODEL_PATH")
    FILE_PATH: str = Field("test_sample/", env="FILE_PATH")
    OUT_FILE: str = Field("results_test.json", env="OUT_FILE")

    @classmethod
    def from_env(cls):
        load_dotenv()
        return cls.model_validate({})
    
if __name__ == "__main__":
    config = AppConfig.from_env()
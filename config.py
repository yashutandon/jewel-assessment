from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

class Settings(BaseSettings):
    # App
    app_name: str = "Jewellery Earring Recommender"
    host: str = "0.0.0.0"
    port: int = 8000
    
    # Paths
    base_dir: Path = BASE_DIR
    images_dir: Path = BASE_DIR / "Jewelry Images"
    csv_path: Path = BASE_DIR / "candidate_dataset.csv"
    template_path: Path = BASE_DIR / "templates" / "index.html"
    cache_dir: Path = BASE_DIR / ".cache"
    
    # ML
    resnet_batch_size: int = 8
    
    # CORS
    allowed_origins: list[str] = ["*"]
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()

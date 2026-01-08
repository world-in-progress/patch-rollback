from pydantic import BaseModel, field_validator

class PatchMeta(BaseModel):
    """Information about the patch of a specific project"""
    name: str
    bounds: tuple[float, float, float, float] # [ min_lon, min_lat, max_lon, max_lat ] 
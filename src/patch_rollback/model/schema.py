from pydantic import BaseModel, field_validator

class GridSchema(BaseModel):
    name: str
    epsg: int
    alignment_origin: tuple[float, float] # [lon, lat], base point of the grid
    grid_info: list[tuple[float, float]] # [(width_in_meter, height_in_meter), ...], grid size in each level
 
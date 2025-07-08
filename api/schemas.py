from pydantic import BaseModel

class Sighting(BaseModel):
    image_id: str
    timestamp: str
    location: str
    detected_text: str


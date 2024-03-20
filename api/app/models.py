from pydantic import BaseModel

class Modelo(BaseModel):
    fixed_acidity:float
    volatile_acidity:float
    citric_acid:float
    residual_sugar:float
    chlorides:float
    free_sulfur_dioxide:float
    total_sulfur_dioxide:float
    density:float
    pH:float
    sulphates:float
    alcohol:float
    
class PredictionResponse(BaseModel):
    worldwide_gross: float
    

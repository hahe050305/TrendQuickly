from fastapi import FastAPI
import pickle
import pandas as pd
from pydantic import BaseModel

# Load trained model
with open("trend_predictor.pkl", "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI app
app = FastAPI()

# Define input schema
class VideoData(BaseModel):
    title: str
    tags: str
    category_id: int
    like_count: int
    comment_count: int

# API route to predict views
@app.post("/predict/")
def predict_views(video: VideoData):
    input_data = pd.DataFrame([video.dict()])
    predicted_views = model.predict(input_data)[0]
    return {"predicted_views": int(predicted_views)}

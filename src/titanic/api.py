from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from titanic.predict import load_model, predict


class Passenger(BaseModel):
    model_config = {"json_schema_extra": {"examples": [
        {"Pclass": 1, "Sex": "female", "SibSp": 0, "Parch": 0}
    ]}}

    Pclass: Literal[1, 2, 3]
    Sex: Literal["male", "female"]
    SibSp: int = Field(ge=0, le=10)
    Parch: int = Field(ge=0, le=10)


class Prediction(BaseModel):
    prediction: int
    probability: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = load_model()  # runs once at startup
    yield


app = FastAPI(title="Titanic survival", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# This is an endpoint
# It is simply a python method that can be called using the API 
@app.get("/health")
def health():
    return {"status": "ok"}

# This is also an endpoint
# This endpoint exposes the predict method which allows apps that consume
# the titanic API to make a prediction about a user-defined passenger.
@app.post("/predict", response_model=Prediction)
def predict_endpoint(passenger: Passenger):
    records = pd.DataFrame([passenger.model_dump()])
    result = predict(app.state.model, records).iloc[0]
    return Prediction(
        prediction=int(result["prediction"]),
        probability=float(result["probability"]),
    )
from typing import List, Annotated
from pydantic import BaseModel, Field, conint


class AdultIncomeRecord(BaseModel):
    age: conint(ge=0, le=120)
    fnlwgt: conint(ge=0)
    capital_gain: conint(ge=0) = Field(alias="capital-gain")
    capital_loss: conint(ge=0) = Field(alias="capital-loss")
    hours_per_week: conint(ge=1, le=99) = Field(alias="hours-per-week")
    workclass: str
    education: str
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    native_country: str = Field(alias="native-country")

    model_config = {
        "populate_by_name": True,
        "extra": "ignore",
    }


class PredictRequest(BaseModel):
    records: Annotated[List[AdultIncomeRecord], Field(min_items=1, max_items=1000)]


class PredictItem(BaseModel):
    prob_1: float
    label: int


class ModelInfo(BaseModel):
    model_name: str
    alias: str
    version: int
    run_id: str
    source: str

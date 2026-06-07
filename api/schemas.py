# filename: api/schemas.py
# purpose:  Pydantic request/response models for the FastAPI serving layer
# version:  1.0

from enum import Enum

from pydantic import BaseModel, ConfigDict


class APIBaseModel(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class Gender(str, Enum):
    male = "Male"
    female = "Female"
    other = "Other"


class Channel(str, Enum):
    email = "Email"
    chat = "Chat"
    phone = "Phone"
    social = "Social media"


class TicketRequest(BaseModel):
    ticket_subject: str
    ticket_description: str
    customer_age: float = 30.0
    customer_gender: Gender = Gender.other
    product_purchased: str = "Unknown"
    ticket_channel: Channel = Channel.email
    days_since_purchase: float | None = None
    response_hour_of_day: float = -1.0


class PredictTypeResponse(APIBaseModel):
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]
    auto_route: bool
    flag_for_review: bool
    model_name: str
    processing_time_ms: float


class PredictPriorityResponse(APIBaseModel):
    predicted_label: str
    confidence: float
    probabilities: dict[str, float]
    model_name: str
    processing_time_ms: float


class PredictResolutionResponse(APIBaseModel):
    predicted_hours: float
    model_name: str
    processing_time_ms: float
    warning: str


class HealthResponse(APIBaseModel):
    status: str
    models_loaded: bool
    model_count: int
    uptime_seconds: float


class ShapFeature(BaseModel):
    feature: str
    shap_value: float


class ExplainPriorityResponse(APIBaseModel):
    predicted_label: str
    top_features: list[ShapFeature]
    model_name: str
    processing_time_ms: float


class ReloadResponse(BaseModel):
    status: str
    reload_time_ms: float

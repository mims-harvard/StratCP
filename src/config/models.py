from typing import Any

from pydantic import BaseModel


class LoggingConfig(BaseModel):
    version: int
    disable_existing_loggers: bool
    formatters: dict[str, Any]
    handlers: dict[str, Any]
    loggers: dict[str, Any]
    root: dict[str, Any]


class HyperparametersConfig(BaseModel):
    learning_rate: float
    batch_size: int
    num_epochs: int

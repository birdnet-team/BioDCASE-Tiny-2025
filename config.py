from pathlib import Path

import yaml
from pydantic import BaseModel, field_validator, Field

from paths import PIPELINE_CONFIG_FILE


class DataPreprocessing(BaseModel):
    audio_slice_duration_ms: int
    sample_rate: int


class FeatureExtraction(BaseModel):
    window_len: int
    window_stride: int
    window_scaling_bits: int
    mel_n_channels: int# = Field(ge=20, le=40, multiple_of=20)
    mel_low_hz: int
    mel_high_hz: int
    mel_post_scaling_bits: int

    @field_validator('mel_high_hz')
    @classmethod
    def validate_mel_high_hz(cls, v, values):
        if v <= values.data['mel_low_hz']:
            raise ValueError(f'mel_high_hz must be strictly greater than mel_low_hz')
        return v


class ModelTraining(BaseModel):
    class EarlyStopping(BaseModel):
        patience: int
    seed: int
    n_epochs: int
    shuffle_buff_n: int
    batch_size: int
    early_stopping: EarlyStopping


class EmbeddedCodeGeneration(BaseModel):
    serial_device: str


class Config(BaseModel):
    data_preprocessing: DataPreprocessing
    feature_extraction: FeatureExtraction
    model_training: ModelTraining
    embedded_code_generation: EmbeddedCodeGeneration


def load_config(config_file: Path = PIPELINE_CONFIG_FILE) -> Config:
    with config_file.open("r") as file:
        yaml_data = yaml.safe_load(file)
    return Config(**yaml_data)

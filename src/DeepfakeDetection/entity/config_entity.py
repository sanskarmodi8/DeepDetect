from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_data: str
    final_data_path: Path
    num_videos: int


@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    data_path: Path
    output_data: Path
    max_frames: int
    fps: int
    resolution: list


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    data_path: Path
    input_shape: list
    batch_size: int
    sequence_length: int
    num_workers: int
    dropout_rate: float
    units: int
    learning_rate: float
    epochs: int
    model_path: Path


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    model_path: Path
    score: Path
    input_shape: list
    batch_size: int
    num_workers: int
    sequence_length: int

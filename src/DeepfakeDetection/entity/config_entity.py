from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    local_data_file: Path
    unzip_dir: Path


@dataclass(frozen=True)
class DataPreprocessingConfig:
    root_dir: Path
    data_path: Path
    output_dir: Path
    max_frames: int
    target_size: list
    min_size: list
    min_neighbors: int
    scale_factor: float
    incremental_save_frequency: int
    scene_change_threshold: float


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path
    train_data_path: Path
    train_labels_path: Path
    val_data_path: Path
    val_labels_path: Path
    model_path: Path
    batch_size: int
    epochs: int
    learning_rate: float
    input_shape: list


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    labels_path: Path
    model_path: Path
    score: Path

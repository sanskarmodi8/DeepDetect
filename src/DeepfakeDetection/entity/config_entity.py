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
    target_size: list
    min_size: list
    min_neighbors: int
    scale_factor: float
    scene_change_threshold: float
    mtcnn_thres: list


@dataclass(frozen=True)
class ModelTrainingConfig:
    const_lr: bool
    root_dir: Path
    train_data_path: Path
    train_labels_path: Path
    val_data_path: Path
    val_labels_path: Path
    model_path: Path
    ckpt_path: Path
    batch_size: int
    epochs: int
    initial_learning_rate: float
    learning_rate_decay: float
    decay_steps: int
    decay_rate: float
    input_shape: list
    pretrained: bool
    num_heads: int
    key_dim: int
    units: int
    activation: str
    dropout_rate: float
    l2: float
    buffer: int
    attention_depth: int


@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    data_path: Path
    labels_path: Path
    model_path: Path
    ckpt_path: Path
    score: Path
    threshold: float
    input_shape: list
    batch_size: int

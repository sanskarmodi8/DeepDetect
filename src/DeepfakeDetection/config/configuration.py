from DeepfakeDetection.constants import CONFIG_FILE_PATH, PARAMS_FILE_PATH
from DeepfakeDetection.entity.config_entity import (
    DataIngestionConfig,
    DataPreprocessingConfig,
    ModelEvaluationConfig,
    ModelTrainingConfig,
)
from DeepfakeDetection.utils.common import create_directories, read_yaml


class ConfigurationManager:
    def __init__(
        self, config_filepath=CONFIG_FILE_PATH, params_filepath=PARAMS_FILE_PATH
    ):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir,
        )

        return data_ingestion_config

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing

        create_directories([config.root_dir])

        data_preprocessing_config = DataPreprocessingConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            output_dir=config.output_dir,
            max_frames=self.params.max_frames,
            target_size=self.params.target_size,
            min_size=self.params.min_size,
            min_neighbors=self.params.min_neighbors,
            scale_factor=self.params.scale_factor,
            scene_change_threshold=self.params.scene_change_threshold,
        )

        return data_preprocessing_config

    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training
        create_directories([config.root_dir])
        model_training_config = ModelTrainingConfig(
            root_dir=config.root_dir,
            train_data_path=config.train_data_path,
            train_labels_path=config.train_labels_path,
            val_data_path=config.val_data_path,
            val_labels_path=config.val_labels_path,
            model_path=config.model_path,
            batch_size=self.params.batch_size,
            epochs=self.params.epochs,
            learning_rate_decay=self.params.learning_rate_decay,
            decay_steps=self.params.decay_steps,
            decay_rate=self.params.decay_rate,
            input_shape=self.params.input_shape,
            pretrained=self.params.pretrained,
            pretrained_top=self.params.pretrained_top,
            include_top=self.params.include_top,
            units=self.params.units,
            activation=self.params.activation,
            dropout_rate=self.params.dropout_rate,
            l2=self.params.l2,
            initial_learning_rate=self.params.initial_learning_rate,
            rotation=self.params.rotation,
            zoom=self.params.zoom,
            contrast=self.params.contrast,
            buffer=self.params.buffer,
            gnoise=self.params.gnoise,
        )

        return model_training_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_directories([config.root_dir])
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            labels_path=config.labels_path,
            model_path=config.model_path,
            score=config.score,
            threshold=self.params.threshold,
        )

        return model_evaluation_config

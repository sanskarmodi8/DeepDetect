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
            source_data=config.source_data,
            final_data_path=config.final_data_path,
            num_videos=self.params.num_videos,
        )

        return data_ingestion_config

    def get_data_preprocessing_config(self) -> DataPreprocessingConfig:
        config = self.config.data_preprocessing

        create_directories([config.root_dir, config.output_data])

        data_preprocessing_config = DataPreprocessingConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            output_data=config.output_data,
            max_frames=self.params.max_frames,
            fps=self.params.fps,
            resolution=self.params.resolution,
        )

        return data_preprocessing_config

    def get_model_training_config(self) -> ModelTrainingConfig:
        config = self.config.model_training
        create_directories([config.root_dir])
        model_training_config = ModelTrainingConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            input_shape=self.params.input_shape,
            batch_size=self.params.batch_size,
            sequence_length=self.params.sequence_length,
            num_workers=self.params.num_workers,
            dropout_rate=self.params.dropout_rate,
            units=self.params.units,
            learning_rate=self.params.learning_rate,
            epochs=self.params.epochs,
            model_path=config.model_path,
        )

        return model_training_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        config = self.config.model_evaluation
        create_directories([config.root_dir])
        model_evaluation_config = ModelEvaluationConfig(
            root_dir=config.root_dir,
            data_path=config.data_path,
            model_path=config.model_path,
            score=config.score,
            input_shape=self.params.input_shape,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
            sequence_length=self.params.sequence_length,
        )

        return model_evaluation_config

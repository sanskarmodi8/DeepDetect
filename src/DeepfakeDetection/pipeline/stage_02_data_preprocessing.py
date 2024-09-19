from DeepfakeDetection import logger
from DeepfakeDetection.components.data_preprocessing import DataPreprocessing
from DeepfakeDetection.config.configuration import ConfigurationManager


class DataPreprocessingPipeline:
    def __init__(self):
        """
        Initializes DataPreprocessingPipeline with a configuration object.

        Args:
            None

        Attributes:
            data_preprocessing_config (DataPreprocessingConfig): Configuration object with paths and settings.
            data_preprocessing (DataPreprocessing): DataPreprocessing component instance.
        """
        config = ConfigurationManager()
        self.data_preprocessing_config = config.get_data_preprocessing_config()
        self.data_preprocessing = DataPreprocessing(
            config=self.data_preprocessing_config
        )

    def main(self):
        """
        Executes the data preprocessing process by calling the run method of DataPreprocessing.
        """
        self.data_preprocessing.run()


STAGE_NAME = "Data Preprocessing"

try:
    logger.info(f"\n\n>>>>>> stage {STAGE_NAME} started <<<<<<\n\n")
    data_preprocessing = DataPreprocessingPipeline()
    data_preprocessing.main()
    logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e

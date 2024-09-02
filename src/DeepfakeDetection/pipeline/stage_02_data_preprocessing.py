from DeepfakeDetection.components.data_preprocessing import DataPreprocessing
from DeepfakeDetection.config.configuration import ConfigurationManager
from DeepfakeDetection import logger


class DataPreprocessingPipeline:
    def __init__(self):
        config = ConfigurationManager()
        self.data_preprocessing_config = config.get_data_preprocessing_config()
        self.data_preprocessing = DataPreprocessing(
            config=self.data_preprocessing_config
        )

    def main(self):
        self.data_preprocessing.execute()


STAGE_NAME = "Data Preprocessing"

try:
    logger.info(f"\n\n>>>>>> stage {STAGE_NAME} started <<<<<<\n\n")
    data_preprocessing = DataPreprocessingPipeline()
    data_preprocessing.main()
    logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e

from DeepfakeDetection import logger
from DeepfakeDetection.components.data_ingestion import DataIngestion
from DeepfakeDetection.config.configuration import ConfigurationManager

STAGE_NAME = "Data Ingestion stage"


class DataIngestionPipeline:
    def __init__(self):
        """
        Initializes DataIngestionPipeline with a configuration object.

        Retrieves the configuration from ConfigurationManager and
        initializes DataIngestion with it.
        """
        config = ConfigurationManager()
        data_ingestion_config = config.get_data_ingestion_config()
        self.data_ingestion = DataIngestion(config=data_ingestion_config)

    def main(self):
        """
        Executes the data ingestion process by calling the run method of DataIngestion.
        """
        self.data_ingestion.run()


if __name__ == "__main__":
    try:
        logger.info(f"\n\n >>>>>> stage {STAGE_NAME} started <<<<<<\n\n")
        obj = DataIngestionPipeline()
        obj.main()
        logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
    except Exception as e:
        logger.exception(e)
        raise e

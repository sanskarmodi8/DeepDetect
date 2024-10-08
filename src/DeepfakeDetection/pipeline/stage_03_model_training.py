from DeepfakeDetection import logger
from DeepfakeDetection.components.model_training import ModelTraining
from DeepfakeDetection.config.configuration import ConfigurationManager

STAGE_NAME = "Model Training stage"


class ModelTrainingPipeline:

    def main(self):
        """
        Execute the model training pipeline by getting the configuration
        from ConfigurationManager, setting up the ModelTraining object and
        calling its execute() method.
        """
        config = ConfigurationManager()
        model_training_config = config.get_model_training_config()
        model_training = ModelTraining(model_training_config)
        # model_training.initialize_mlflow()  # Initialize mlflow logging
        model_training.execute()


try:
    logger.info(f"\n\n >>>>>> stage {STAGE_NAME} started <<<<<<\n\n")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e

from DeepfakeDetection import logger
from DeepfakeDetection.components.model_evaluation import ModelEvaluation
from DeepfakeDetection.config.configuration import ConfigurationManager


class ModelEvaluationPipeline:
    def __init__(self):
        config = ConfigurationManager()
        self.model_evaluation_config = config.get_model_evaluation_config()
        self.model_evaluation = ModelEvaluation(config=self.model_evaluation_config)

    def main(self):
        self.model_evaluation.execute()


STAGE_NAME = "Model Evaluation stage"

try:
    logger.info(f"\n\n>>>>>> stage {STAGE_NAME} started <<<<<<\n\n")
    model_evaluation_pipeline = ModelEvaluationPipeline()
    model_evaluation_pipeline.main()
    logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e

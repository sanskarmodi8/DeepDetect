from dotenv import load_dotenv

from DeepfakeDetection import logger
from DeepfakeDetection.pipeline.stage_01_data_ingestion import DataIngestionPipeline
from DeepfakeDetection.pipeline.stage_02_data_preprocessing import (
    DataPreprocessingPipeline,
)
from DeepfakeDetection.pipeline.stage_03_model_training import ModelTrainingPipeline
from DeepfakeDetection.pipeline.stage_04_model_evaluation import ModelEvaluationPipeline

# load the env variables for the mlflow tracking
load_dotenv()

# STAGE_NAME = "Data Ingestion stage"
# try:
#     logger.info(f"\n\n>>>>>> stage {STAGE_NAME} started <<<<<<\n\n")
#     data_ingestion = DataIngestionPipeline()
#     data_ingestion.main()
#     logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
# except Exception as e:
#     logger.exception(e)
#     raise e

# STAGE_NAME = "Data Preprocessing"

# try:
#     logger.info(f"\n\n>>>>>> stage {STAGE_NAME} started <<<<<<\n\n")
#     data_preprocessing = DataPreprocessingPipeline()
#     data_preprocessing.main()
#     logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
# except Exception as e:
#     logger.exception(e)
#     raise e

STAGE_NAME = "Model Training stage"

try:
    logger.info(f"\n\n >>>>>> stage {STAGE_NAME} started <<<<<<\n\n")
    obj = ModelTrainingPipeline()
    obj.main()
    logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e

STAGE_NAME = "Model Evaluation stage"

try:
    logger.info(f"\n\n>>>>>> stage {STAGE_NAME} started <<<<<<\n\n")
    model_evaluation_pipeline = ModelEvaluationPipeline()
    model_evaluation_pipeline.main()
    logger.info(f"\n\n>>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
except Exception as e:
    logger.exception(e)
    raise e

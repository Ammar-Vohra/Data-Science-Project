from src.DataScience import logger
from src.DataScience.pipeline.data_ingestion_pipeline import DataIngestionTrainingPipeline
from src.DataScience.pipeline.data_validation_pipeline import DataValidationTrainingPipeline
from src.DataScience.pipeline.data_transformation_pipeline import DataTransformationTrainingPipeline
from src.DataScience.pipeline.model_training_pipeline import ModelTrainingPipeline





STAGE_NAME = "Data Ingestion Stage"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<")
    data_ingestion = DataIngestionTrainingPipeline()
    data_ingestion.initiate_data_ingestion()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=======x")
except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = "Data Validation Stage"
try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<")
    data_validation = DataValidationTrainingPipeline()
    data_validation.initiate_data_validation()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=======x")
except Exception as e:
    logger.exception(e)
    raise e


STAGE_NAME = "Data Transformation Stage"

try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<")
    data_transformation = DataTransformationTrainingPipeline()
    data_transformation.initiate_data_transformation()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=======x")
except Exception as e:
    logger.exception(e)
    raise e



STAGE_NAME = "Model Training Stage"

try:
    logger.info(f">>>>>>> stage {STAGE_NAME} started <<<<<<")
    model_trainer = ModelTrainingPipeline()
    model_trainer.initiate_model_training()
    logger.info(f">>>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx=======x")
except Exception as e:
    logger.exception(e)
    raise e
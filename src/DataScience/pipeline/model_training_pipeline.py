from src.DataScience.config.configuration import ConfigurationManager
from src.DataScience.components.model_training import ModelTrainer
from src.DataScience import logger
from pathlib import Path

STAGE_NAME = "Model Training Stage"


class ModelTrainingPipeline:
    def __init__(self):
        pass

    def initiate_model_training(self):
            config = ConfigurationManager()
            model_trainer_config = config.get_model_trainer_config()
            model_trainer = ModelTrainer(config=model_trainer_config)
            model_trainer.train()
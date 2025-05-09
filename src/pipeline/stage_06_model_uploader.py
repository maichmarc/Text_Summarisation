from src.config.configuration import ConfigurationManager
from src.components.model_uploader import ModelUploader
from src.my_logging.my_logger import logging

class ModelUploaderTrainingPipeline:
    def __init__(self):
        pass

    def main(self):    
        config = ConfigurationManager()
        model_uploader_config = config.get_model_uploader_config()
        model_uploader = ModelUploader(config=model_uploader_config)
        model_uploader.upload_model()
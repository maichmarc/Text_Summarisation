import torch
import os
from huggingface_hub import HfApi, HfFolder
from transformers import TrainingArguments, Trainer
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import PegasusTokenizerFast
from transformers import PegasusForConditionalGeneration
from src.config.configuration import ModelUploaderConfig

class ModelUploader:
    def __init__(self, config: ModelUploaderConfig):
        self.config = config
        
    def upload_model(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path, local_files_only=True)
        model_pegasus = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path, local_files_only=True).to(device)
        # tokenizer = PegasusTokenizerFast.from_pretrained(self.config.tokenizer_path)
        # model_pegasus = PegasusForConditionalGeneration.from_pretrained(self.config.model_path)

        api = HfApi()

        token = HfFolder.get_token()

        repo_name = 'maichmarc/textS'

        api.create_repo(repo_id=repo_name, token=token)

        model_pegasus.push_to_hub(repo_name)
        tokenizer.push_to_hub(repo_name)
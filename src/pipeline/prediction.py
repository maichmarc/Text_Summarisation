from src.config.configuration import ConfigurationManager
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self, text):
        # tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        tokenizer = AutoTokenizer.from_pretrained('maichmarc/textS')
        model = AutoModelForSeq2SeqLM.from_pretrained("maichmarc/textS")

        gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}
        # pipe = pipeline("summarization", model=self.config.model_path, tokenizer=tokenizer)
        pipe = pipeline("summarization", model = model, tokenizer=tokenizer)

        print('Dialogue:')
        print(text)
        output = pipe(text, **gen_kwargs)[0]["summary_text"]

        print("\nModel Summary:") 
        print(output)

        return output

        
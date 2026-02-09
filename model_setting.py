from .model import PretrainedModel, FineTuningModel
from .preprocessing import FineTuningDataset
from transformers import PreTrainedTokenizerFast
from .utility.dataset_processor_tld import wrap_tld
import torch

class DomainClassifier:
    def __init__(self, model_path='/srv/projects/surf/SURF_AI_model/finetuning_0120_1528.pt'):
        self.device = torch.device('cpu')
        self.tokenizer = PreTrainedTokenizerFast(tokenizer_file='/srv/projects/surf/SURF_AI_model/tokenizer-2-32393-both-tld.json')
        self.pt_model_c = PretrainedModel(2273, 256, 8, 768, 12, 82)
        self.pt_model_t = PretrainedModel(32393, 256, 8, 768, 12, 35)
        self.ft_model = FineTuningModel(self.pt_model_t, self.pt_model_c)

        self.processor = FineTuningDataset(df=None, tokenizer=self.tokenizer, max_len_t=35, max_len_c=82)

        self.ft_model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.ft_model.eval()
        print(f"✅ Real model loaded from {model_path}")

    def predict(self, domain) :
        processed_domain = wrap_tld(domain)

        x_token = self.processor.domain_to_token(processed_domain)
        x_char = self.processor.domain_to_ids(processed_domain)

        x_token_tensor = torch.from_numpy(x_token).unsqueeze(0).to(torch.long).to(self.device)
        x_char_tensor = torch.from_numpy(x_char).unsqueeze(0).to(torch.long).to(self.device)

        with torch.no_grad() :
            logits = self.ft_model(x_token_tensor, x_char_tensor)
            pred = torch.argmax(logits, dim=1).item()
            probs = torch.softmax(logits, dim=1)

        print(f"🔍 [DNS Query] {domain} ({processed_domain}) -> Prediction: {pred}")
        return pred, probs

if __name__ == '__main__':
    classifier = DomainClassifier()
    classifier.predict("google.com")
    # classifier.predict("gdheklhhsspojpiqjkre.com")
    classifier.predict("ubuntu.com")

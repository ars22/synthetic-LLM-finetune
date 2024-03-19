import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


class LLama2:
    def __init__(self):
        super().__init__() 
        self.config = {
            'model_id': 'NousResearch/Llama-2-7b-hf',
            'flash_attention2': True,
        }
        print("loading weights from pretrained Llama2: %s" % self.config['model_id'])
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model_id'], 
            use_flash_attention_2=self.config["flash_attention2"], 
            torch_dtype=torch.bfloat16, 
            device_map='balanced',
            trust_remote_code = True)
        

if __name__ == "__main__":
    llama2 = LLama2()
    llama2.model.eval()
    text = "Hello my name is"
    idx = torch.tensor(llama2.tokenizer.encode(text), dtype=torch.int32).unsqueeze(0).to(llama2.model.device)
    out = llama2.model.generate(idx, max_new_tokens=54, top_k=1)
    print(llama2.tokenizer.decode(out.squeeze()))

import torch
from transformers import BertModel, BertTokenizer


class OurBert(BertModel):
    def __init__(self, config):
        super().__init__(config)

    def forward(self, input_ids, attention_mask, token_type_ids):
        out = super().forward(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, output_hidden_states=True)
        logits = torch.cat(out["hidden_states"][-3:-2], -1).squeeze(0)
        return logits


model = OurBert.from_pretrained("/home/shmyrev/kaldi/egs/ac/ru-tts/Matcha/Matcha-TTS/rubert-base")
tokenizer = BertTokenizer.from_pretrained("/home/shmyrev/kaldi/egs/ac/ru-tts/Matcha/Matcha-TTS/rubert-base")
dummy_model_input = tokenizer("Привет, как дела!", return_tensors="pt")

# export
torch.onnx.export(
    model, 
    tuple(dummy_model_input.values()),
    f="bert-ru.onnx",  
    input_names=['input_ids', 'attention_mask', 'token_type_ids'], 
    output_names=['logits'], 
    dynamic_axes={'input_ids': {0: 'batch_size', 1: 'sequence'}, 
                  'attention_mask': {0: 'batch_size', 1: 'sequence'}, 
                  'token_type_ids': {0: 'batch_size', 1: 'sequence'}, 
                  'logits': {0: 'batch_size', 1: 'sequence'}}, 
    do_constant_folding=True, 
    opset_version=17, 
)

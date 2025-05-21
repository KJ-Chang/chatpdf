import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(model_name: str, device: str = 'cpu'):
    """
    load model and tokenizer from huggingface hub

    Args:
        model_name (str): Model name or path
        device (str): Device to load the model on ('cuda' or 'cpu')

    Returns:
        tuple: (model, tokenizer)
    """
    if device == 'cuda':
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)

    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer

def generate(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, 
                       truncation=True,
                        padding=True,
                        return_attention_mask=True,
                        return_tensors='pt'
                       ).to(device)
     
    with torch.no_grad():
        outputs= model.generate(
            input_ids = inputs.input_ids, 
            attention_mask = inputs.attention_mask,
            pad_token_id = tokenizer.pad_token_id,
            num_beams=3,
            max_new_tokens=258, 
            do_sample=True, 
            # top_k=30, 
            top_p=0.9, 
            temperature=0.7,
            repetition_penalty=1.8,
            early_stopping=True
            )
        
        prompt_length = inputs.input_ids.shape[1]
        response = tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)

    return response
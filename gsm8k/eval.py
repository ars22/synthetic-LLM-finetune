import transformers
import torch
import os
import json
import random
import numpy as np
import argparse
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
import datasets
from dataset import get_examples, GSMDataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
import sys
import re
from rouge_score import rouge_scorer
from collections import defaultdict
import re


def remove_extra_spaces(string):
    # Remove extra spaces using regular expression
    cleaned_string = re.sub(' +', ' ', string)
    return cleaned_string


def answer_cleansing(pred):
    pred=pred.replace(",", "")
    pred = [delete_extra_zero(s.replace(",", "")) for s in re.findall(r'-?\d+/?\.?\d*', pred)]
    # If there is no candidate in list, null is set.
    if len(pred) == 0:
        pred = ""
    else:
        pred = pred[-1]
    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred != "":
        if pred[-1] == ".":
            pred = pred[:-1]
        if pred[-1] == "/":
            pred = pred[:-1]
    return pred



def delete_extra_zero(n):
    try:
        n=float(n)
    except:
        # print("None {}".format(n))
        return n
    if isinstance(n, int):
        return str(n)
    if isinstance(n, float):
        n = str(n).rstrip('0')  
        n = int(n.rstrip('.')) if n.endswith('.') else float(n)  
        n=str(n)
        return n
    

def get_collate_fn(tokenizer) -> Callable[[List[Dict]], Dict[str, Union[List, torch.Tensor]]]:
    """Returns a collate function for the given tokenizer.
       The collate function takes a list of examples (dicts, where values are lists of
       ints [tokens] or strings [the original texts]) and returns a batch of examples,
       PyTorch tensors padded to the maximum length. Strings are passed through."""
    def collate_fn(batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k.endswith('input_ids') or k.endswith('attention_mask') or k.endswith('labels'):
                to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                if k.endswith('input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('attention_mask'):
                    padding_value = 0
                else:
                    raise ValueError(f"Unexpected key in batch '{k}'")
                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                padded_batch[k] = padded_batch[k].flip(dims=[1])
            else:
                padded_batch[k] = [ex[k] for ex in batch]
        # print(f"Returning: {padded_batch}, {tokenizer.padding_side}")
        return padded_batch
    return collate_fn



def tokenize_batch_element(prompt: str, tokenizer  ) -> Dict:
    """Tokenize a single batch element.
       We also create the labels for the input, which is of length equal to
         the length of the prompt + response with -100 for the prompt tokens.
    """
    prompt_tokens = tokenizer(prompt, add_special_tokens=False)
    return prompt_tokens


def get_batch_iterator(tokenizer,
                       split: str = 'train',
                       chat_template: Optional[str] = None,
                       batch_size: int = 1) -> Iterator[Dict]:
    """Get an iterator over batches of data. Stops after 1 epoch.
    """
    
    flat_data = get_examples(split)
    collate_fn = get_collate_fn(tokenizer)
    batch = []
    for input in flat_data:
        ground_truth = delete_extra_zero(input["answer"].split("#### ")[-1].replace(",", "").replace("<|endoftext|>", ""))
        if chat_template is not None:
            prompt = chat_template.format(input["question"].strip())
        else:
            prompt = input['question'].strip()
        batch_element = tokenize_batch_element(prompt, tokenizer)
        batch_element['gt_answer'] = ground_truth
        batch_element['answer'] = input["answer"].strip()
        batch_element['question'] = input['question'].strip()
        batch_element['prompt'] = tokenizer.decode(batch_element['input_ids'], skip_special_tokens=True)
        batch.append(batch_element)
        if len(batch) == batch_size:
            yield collate_fn(batch)
            batch = []


def dump_files(responses, all_models, temp):
    json_out = responses
    with open(all_models[temp], 'w') as f:
        json.dump(json_out, f, indent=2)
    print('dumped to file')

def main():

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # fetch model and tokenizer
    if args.model_name == 'llama2-7b':
        model_path = 'NousResearch/Llama-2-7b-hf'
        chat_template = "<s> {}"
    elif args.model_name == 'llama2-7b-chat':
        model_path = 'NousResearch/Llama-2-7b-chat-hf'
        chat_template = "<s>[INST] {} End your answer as, 'Hence the answer is: '. [/INST]"
    elif args.model_name == 'mistral7b':
        model_path = 'mistralai/Mistral-7B-v0.1'
        chat_template = "<s> {}"
    elif args.model_name == 'mistral7b-chat':
        model_path = 'mistralai/Mistral-7B-Instruct-v0.1'
        chat_template = "<s>[INST] {} End your answer as, 'Hence the answer is: '. [/INST]"
    elif args.model_name == 'gemma-7b':
        model_path = 'google/gemma-7b'
        chat_template = "{}"
    elif args.model_name == 'gemma-7b-chat':
        model_path = 'google/gemma-7b-it'
        chat_template = "<bos><start_of_turn>user\n {} <end_of_turn>\n<start_of_turn>model\n"
    policy = transformers.AutoModelForCausalLM.from_pretrained(model_path, device_map='balanced')
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # setup output directory
    all_models = {}
    temps = [float(t) for t in args.temperatures.split(',')]
    output_dir = os.path.join(
        args.model_name + '_samples', f'gsm8k_split_{args.split}_maxlen_{args.max_length}')
    os.makedirs(output_dir, exist_ok=True)
    for temp in temps:
        all_models[temp] = os.path.join(output_dir, f'temp_{temp}.json')

    # scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    # loop over temperatures
    for temp in temps:
        print(f'Generating samples at temperature {temp}')
        policy.eval()
        policy.half()
        batch_iterator = get_batch_iterator(tokenizer=tokenizer, 
            split='test', batch_size=args.batch_size, chat_template=chat_template)
        batch_idx = 0
        responses = defaultdict(list) 
        # loop over batches
        for batch in tqdm(batch_iterator):
            batch_idx += 1
            
            # construct input
            generator_input = {'input_ids': batch['input_ids'].to('cuda'),
                               'attention_mask': batch['attention_mask'].to('cuda'),}

            with torch.no_grad():
                if temp > 0.0:
                    outputs = policy.generate(**generator_input, max_new_tokens=args.max_length, do_sample=True, top_p=0.9, temperature=temp, pad_token_id=tokenizer.pad_token_id)
                else:
                    # do greedy decoding
                    outputs = policy.generate(**generator_input, max_new_tokens=args.max_length, do_sample=False, pad_token_id=tokenizer.pad_token_id)
                
                for idx, output in enumerate(outputs):
                    responses["question"].append(batch['question'][idx])
                    responses["answer"].append(batch['answer'][idx])
                    responses["gt_answer"].append(batch['gt_answer'][idx])
                    responses["prompt"].append(batch['prompt'][idx])
                    response = tokenizer.decode(output, skip_special_tokens=True)
                    response = remove_extra_spaces(response)
                    response = response.replace(batch['prompt'][idx], "")
                    responses["response"].append(response)
                    responses["cleansed_response"].append(answer_cleansing(response))
                    responses["rouge1"].append(scorer.score(batch['answer'][idx], response)['rouge1'].fmeasure)
                    responses["rougeL"].append(scorer.score(batch['answer'][idx], response)['rougeL'].fmeasure)

            print(f'finished generating {batch_idx * args.batch_size} prompts')
            print(f'current accuracy {(np.array(responses["gt_answer"]) == np.array(responses["cleansed_response"])).mean()}')
            print(f'current rouge1 {np.array(responses["rouge1"]).mean()}')
            print(f'current rougeL {np.array(responses["rougeL"]).mean()}')
            dump_files(responses, all_models, temp)

        dump_files(responses, all_models, temp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='llama7b')
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--temperatures', type=str, default='0.7')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=1024)
    args = parser.parse_args()

    main()





# import torch as th
# from dataset import get_examples, GSMDataset
# from calculator import sample
# from transformers import GPT2Tokenizer, GPT2LMHeadModel
# from transformers import AutoModelForCausalLM, AutoTokenizer
# import argparse
# from tqdm import tqdm
# import json


# MODEL_MAP = {
#     'mistral-7b-instruct': 'mistralai/Mistral-7B-Instruct-v0.1',
# }

# TOKENIZER_MAP = {
#     'mistral-7b-instruct': 'mistralai/Mistral-7B-Instruct-v0.1',
# }


# parser = argparse.ArgumentParser(description="Grade School Math Evaluation")
# parser.add_argument("--model", type=str, default="gpt2", help="Model name or path")
# parser.add_argument("--batchsize", type=int, default=4, help="Batch size for evaluation")
# parser.add_argument("--max_length", type=int, default=512, help="Maximum length of input sequences")


# args = parser.parse_args()

# model_name = args.model
# batch_size = args.batchsize

# device = "cuda" if th.cuda.is_available() else "cpu"

# model = AutoModelForCausalLM.from_pretrained(MODEL_MAP[model_name])
# tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MAP[model_name])

# model.to(device)
# print("Model Loaded")
# test_examples = get_examples("test")
# decoded_outputs = {}
    
# for i in tqdm(range(0, len(test_examples), batch_size)):
#     batch = test_examples[i:i+batch_size]
#     batch = [{"role": "user", "content": x["question"]} for x in batch]
#     encodeds = tokenizer.apply_chat_template(batch, return_tensors="pt", max_length=args.max_length)
#     model_inputs = encodeds.to(device)
#     generated_ids = model.generate(model_inputs, max_new_tokens=1000, do_sample=False)
#     decoded = tokenizer.batch_decode(generated_ids)
#     for j in range(i, i+batch_size):
#         decoded_outputs[j] = {
#             "input": test_examples[j],
#             "output": decoded[j-i]
#         }

# json.dump(decoded_outputs, open(f"outputs_{model_name}.json", "w"))
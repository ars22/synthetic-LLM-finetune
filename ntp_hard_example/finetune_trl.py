import argparse
from contextlib import nullcontext
import os
import torch
from tqdm import tqdm

from data import get_dataset
from utils.training_utils import get_lr, get_run_name, AverageMeter
from torch.utils.data import DataLoader
from tokenizing import get_tokenizer
from transformers import  AutoModelForCausalLM, GenerationConfig, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
from pathlib import Path
from data.graphs import prefix_target_list
from transformers import AutoTokenizer
from torch.nn.utils.rnn import pad_sequence
from peft import PeftModel
import evaluate
import numpy as np
from accelerate import Accelerator
from transformers import TrainerCallback 
from trl import SFTTrainer


# some utils
def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    outputs, labels = eval_pred
    # print("o,l", outputs.shape, labels.shape)
    # references = labels[:, 1:]
    # predictions = np.argmax(logits[:, :-1], axis=-1)[references !=-100].flatten()
    # references = references[references!=-100].flatten()
    # np.argmax(logits, axis=-1)[labels != -100].flatten()
    predictions = outputs[labels!=-100].flatten()
    references = labels[labels!=-100].flatten()
    return metric.compute(predictions=predictions, references=references)

def get_batch_loss(output, labels):
    shifted_labels = labels[..., 1:].contiguous()
    output = output[..., :-1, :].contiguous()

    loss_function = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction='mean')
    # get the mean loss for each sequence in a batch
    loss = loss_function(output.transpose(-1,-2), shifted_labels)

    return loss

# Trainer class
class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
        logits = outputs['logits']
        loss = get_batch_loss(logits, inputs['labels'])
        # loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        # print(logits[:, :-1, :].shape, inputs['labels'][:, 1:].shape)
        # loss = loss_fn(logits[:, :-1, :], inputs['labels'][:, 1:])
        # outputs = model(inputs['input_ids'], labels=inputs['labels'], attention_mask=inputs['attention_mask'])
        # return (outputs['loss'], outputs['logits']) if return_outputs else outputs['loss']
        return (loss, logits) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only: bool, ignore_keys=None):
        # forward pass
        with torch.no_grad():
            labels = inputs['labels']
            generation = model.generate(inputs['prompt_input_ids'], temperature=100., top_k=1, max_new_tokens=inputs['answer_input_ids'].shape[1], 
                                        pad_token_id=self.tokenizer.pad_token_id, attention_mask=(inputs['prompt_input_ids']!=self.tokenizer.pad_token_id).long(), do_sample=True)
            ntp_outputs = model(inputs['input_ids'], attention_mask=inputs['attention_mask'])
            loss = get_batch_loss(ntp_outputs['logits'], labels)
        # return (outputs['loss'], outputs['logits'], inputs['labels'])
        # print(generation.shape, labels.shape)
        return (loss, generation, labels)


    

# Dataset wrapper
class GraphsDataset(torch.utils.data.Dataset):   
    def __init__(self, graphs, tokenizer):
        self.graphs = graphs
        self.tokenizer = tokenizer
    def __getitem__(self, idx):
        graph, path = self.graphs[idx]
        tokenized_graph = self.tokenizer(graph)
        tokenized_path = self.tokenizer(path)
        tokenized = {
            'input_ids': tokenized_graph['input_ids'] + tokenized_path['input_ids'], 
            'attention_mask': tokenized_graph['attention_mask'] + tokenized_path['attention_mask'], 
            'labels': len(tokenized_graph['input_ids']) * [-100] + tokenized_path['input_ids'],
            'prompt_input_ids': tokenized_graph['input_ids'],
            'answer_input_ids': tokenized_path['input_ids'],   
        }
        return tokenized
    def __len__(self):
        return len(self.graphs)
    

# collate function
def get_collate_fn(tokenizer):
    def collate_fn(batch):
        # first, pad everything to the same length
        padded_batch = {}
        for k in batch[0].keys():
            if k in ['input_ids', 'attention_mask', 'labels']:
                to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                if k.endswith('input_ids'):
                    padding_value = tokenizer.pad_token_id
                elif k.endswith('labels'):
                    padding_value = -100
                elif k.endswith('attention_mask'):
                    padding_value = 0
                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                padded_batch[k] = padded_batch[k].flip(dims=[1])
            elif k == 'prompt_input_ids':
                to_pad = [torch.LongTensor(ex[k][::-1]) for ex in batch]
                padding_value = tokenizer.pad_token_id
                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
                padded_batch[k] = padded_batch[k].flip(dims=[1])
            elif k == 'answer_input_ids':
                to_pad = [torch.LongTensor(ex[k]) for ex in batch]
                padding_value = -100
                padded_batch[k] = pad_sequence(to_pad, batch_first=True, padding_value=padding_value)
            else:
                raise ValueError(f"Unexpected key in batch '{k}'")
        return padded_batch
    return collate_fn

# Parse arguments
parser = argparse.ArgumentParser(description="Next-token failures")
# Data
parser.add_argument(
    "--model", default='llama2', type=str, help="Type of model"
    )
parser.add_argument(
    "--dataset", default='graph', type=str, help="Choice of dataset"
    )
parser.add_argument(
    "--n_train", default=20000, type=int, help="Number of training samples"
    )
parser.add_argument(
    "--n_test", default=1000, type=int, help="Number of test samples"
    )
parser.add_argument(
    "--num_nodes", default=50, type=int, help="Number of node values in graph"
    )
parser.add_argument(
    "--deg", default=2, type=int, help="Degree of starting node"
    )
parser.add_argument(
    "--path_len", default=5, type=int, help="Path length in star graph"
    )
parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size",
    )
parser.add_argument(
        "--lr", type=float, default=5e-4, help="Learning rate",
    )
parser.add_argument(
        "--weight_decay", type=float, default=1e-2, help="Strength of weight decay",
    )
parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs",
    )
parser.add_argument(
        "--save_dir", type=str, default='runs_llama2', help="Save directory",
    )

args = parser.parse_args()

# System stuff
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
if os.environ.get('LOCAL_RANK') is not None:
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        device_map = {'': local_rank}
os.environ["WANDB_DISABLED"] = "true"
dtype = 'bfloat16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]


# create a logging and saving directory
Path(args.save_dir).mkdir(parents=True, exist_ok=True)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "NousResearch/Llama-2-7b-hf", 
    trust_remote_code=True)
tokenizer.pad_token_id = tokenizer.eos_token_id

# Data stuff
data_path = './data/datasets/graphs/deg_' + str(args.deg) + '_path_' + str(args.path_len) + '_nodes_' + str(args.num_nodes)
train_path, test_path = data_path + '_train_200000.txt', data_path + '_test_20000.txt'
train_graphs = prefix_target_list(train_path, reverse=True)[:args.n_train]
test_graphs = prefix_target_list(test_path, reverse=True)[:args.n_test]
train_data = GraphsDataset(train_graphs, tokenizer)
test_data = GraphsDataset(test_graphs, tokenizer)
train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=get_collate_fn(tokenizer))
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, collate_fn=get_collate_fn(tokenizer))

# save path
run_name = get_run_name(args)
path = './checkpoints/' + run_name + '.pt'

# model stuff
model = AutoModelForCausalLM.from_pretrained(
    'NousResearch/Llama-2-7b-hf', load_in_8bit=True, device_map={"": Accelerator().process_index})
model.train()
config = LoraConfig(
    r=8, 
    lora_alpha=32, 
    target_modules=find_all_linear_names(model), 
    lora_dropout=0.05,
    bias="none", 
    task_type="CAUSAL_LM"
)

# fetch trainer
gradient_accumulation_steps = 1
num_devices = 4
max_steps = int(args.epochs*len(train_data))//(args.batch_size*gradient_accumulation_steps*num_devices)
steps_per_epoch = len(train_data)//(args.batch_size*gradient_accumulation_steps*num_devices)
print(f"steps_per_epoch:{steps_per_epoch} max_steps: {max_steps}")


training_args = TrainingArguments(
    per_device_train_batch_size=args.batch_size//num_devices,
    per_device_eval_batch_size=args.batch_size//num_devices,
    gradient_accumulation_steps=gradient_accumulation_steps,
    warmup_steps=max(1, max_steps//10),
    max_steps=max_steps,
    learning_rate=args.lr,
    bf16=True,
    bf16_full_eval=True,
    logging_steps=max(1, max_steps//20),
    logging_dir=f'{args.save_dir}/logs',
    optim="adamw_hf",
    save_steps=steps_per_epoch,
    save_only_model=False,
    ddp_find_unused_parameters= False,
    deepspeed='ds_config.json',
    weight_decay=args.weight_decay,
    report_to='none',
    output_dir=args.save_dir,
    eval_steps=steps_per_epoch,
    evaluation_strategy="steps",
    remove_unused_columns=False
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=test_data,
    peft_config=config,
    data_collator=get_collate_fn(tokenizer),
    compute_metrics=compute_metrics,
    packing=True
)

trainer.train()

# class CustomCallback(TrainerCallback):
    
#     def __init__(self, trainer) -> None:
#         super().__init__()
#         self._trainer = trainer
    
#     def on_epoch_end(self, args, state, control, **kwargs):
#         control_copy = deepcopy(control)
#         self._trainer.evaluate(eval_dataset=self._trainer.train_dataset, metric_key_prefix="train")
#         return control_copy


# trainer = CustomTrainer(
#         model=model,
#         train_dataset=train_data,
#         eval_dataset=test_data,
#         args=training_args,
#         compute_metrics=compute_metrics,
#         data_collator=get_collate_fn(tokenizer),
#         tokenizer=tokenizer
#     )

# trainer.add_callback(CustomCallback(trainer)) 


# model.save_pretrained(args.save_dir)

# new_model = AutoModelForCausalLM.from_pretrained(
#     'NousResearch/Llama-2-7b-hf', 
#     use_flash_attention_2=True, 
#     torch_dtype=torch.bfloat16, 
#     trust_remote_code = True,
#     do_sample=True,
#     temperature=1000
# )

# peft_model_id = f"{args.save_dir}/checkpoint-{max_steps}"
# peft_model = PeftModel.from_pretrained(new_model, peft_model_id)

# new_model.save_pretrained(args.save_dir)
# tokenizer.save_pretrained(args.save_dir)


# results = {}
# num_iters = 0
# for ep in range(args.epochs):
#     train_bar = tqdm(train_loader)
#     total_loss, total_acc = AverageMeter(), AverageMeter()
#     for x in train_bar:
#         if num_iters % args.save_every == 0 and num_iters > 0:
#             torch.save(
#                 model.state_dict(),
#                 path + "_epoch_" + str(ep)
#             )
#         # determine and set the learning rate for this iteration
#         lr = get_lr(num_iters, args.lr, warmup_iters, lr_decay_iters, min_lr) if decay_lr else args.lr
#         for param_group in optimizer.param_groups:
#             param_group['lr'] = lr

#         # with ctx:
#             # logits, loss, accs = model(x)
#         with ctx:
#             x_hat = x.clone()
#             x_hat[x==-1] = tokenizer.pad_token_id  
#             logits = model(x_hat)['logits']
#             output = logits[:, :-1]
#             shifted_labels = x_hat[:, 1:]
#             loss_function = torch.nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, reduction='mean')
#             loss = loss_function(output.reshape(-1, output.size(-1)), shifted_labels.reshape(-1))
#             accs = {'acc': (output.argmax(-1) == shifted_labels)[shifted_labels!=tokenizer.pad_token_id].float().mean().item()}
#         total_loss.update(loss.item(), (x_hat!=tokenizer.pad_token_id).sum())
#         total_acc.update(accs['acc'], (x_hat!=tokenizer.pad_token_id).sum())
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         optimizer.zero_grad(set_to_none=True)
#         num_iters += 1
#         train_bar.set_description(
#             'Epoch: [{}/{}] Loss: {:.4f} Acc: {:.2f}'.format(ep, args.epochs, total_loss.get(),
#              total_acc.get(percentage=True))
#         )

        # evaluate the loss on train/val sets and write checkpoints
        # if num_iters % args.eval_every == 0 and num_iters > 1:
        #     # Generate sequences and check accuracies
        #     if args.eval_train:
        #         results = evaluate(model, train_loader, temperature=0.8, top_k=top_k, results=results, mode='Train')
        #         results = evaluate_forced(model, train_loader, results=results, mode='train')

        #     results = evaluate(model, test_loader, temperature=0.8, ctx=ctx, top_k=top_k, results=results, mode='Test')
        #     results = evaluate_forced(model, test_loader, ctx=ctx, results=results, mode='Test')




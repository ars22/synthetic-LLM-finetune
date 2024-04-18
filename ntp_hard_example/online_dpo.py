import argparse
from contextlib import nullcontext
import torch
from tqdm import tqdm

from data import get_dataset
from utils.training_utils import get_lr, get_run_name, AverageMeter
from torch.utils.data import DataLoader
from models import get_model
from tokenizing import get_tokenizer
import wandb
from pprint import pprint
import numpy as np
from datasets import Dataset
from copy import deepcopy
from transformers import TrainingArguments
from trl import DPOTrainer
from transformers import AutoTokenizer
import os

def generate_and_score(model, loader, ctx, temperature, top_k, n_samples=1):
    num_prefix_tokens = loader.dataset.num_prefix_tokens
    num_target_tokens = loader.dataset.num_target_tokens
    # Switch dataset and model to "eval" mode
    loader.dataset.eval()
    model.eval()
    bar = tqdm(loader)
    #model.set_cache(loader.dataset.device)
    x_dataset = []
    y_pred_dataset = []
    correct_dataset = []
    for x in bar:
        y = x[:, num_prefix_tokens:].clone()
        x = x[:, :num_prefix_tokens].clone()
        y = y
        y_pred = []
        for i in range(n_samples):
            with ctx:
                y_pred.append(
                    model.generate(x, min_length=num_target_tokens, 
                                   max_new_tokens=num_target_tokens, temperature=temperature, top_k=top_k,
                                   do_sample=True, attention_mask = torch.ones_like(x), pad_token_id=2))
        y_pred = torch.stack(y_pred, dim=0).transpose(0, 1)
        # y_pred should be bs x nsamples x length
        correct = torch.stack([y.eq(y_pred[:, i, -num_target_tokens:]).float().mean(dim=1) for i in range(n_samples)]).transpose(0, 1)
        y_pred_dataset.append(y_pred[:, :, -num_target_tokens:].cpu())
        correct_dataset.append(correct.cpu())
        x_dataset.append(x.cpu())
        # correct should be bs x nsamples
    # Switch back to train mode
    loader.dataset.train()
    model.train()
    y_pred_dataset = torch.cat(y_pred_dataset, dim=0)
    x_dataset = torch.cat(x_dataset, dim=0)
    scores = torch.cat(correct_dataset, dim=0)
    generations = np.array(tokenizer.batch_decode(y_pred_dataset.reshape(-1, y_pred_dataset.shape[-1]))).reshape(*y_pred_dataset.shape[:2])
    prompts = np.array(tokenizer.batch_decode(x_dataset))
    return prompts, generations, scores


def evaluate(model, loader, ctx, temperature, top_k, results=None, mode='test', pass_at_k=1):
    """
    Generates sequences (without teacher-forcing) and calculates accuracies
    """
    num_prefix_tokens = loader.dataset.num_prefix_tokens
    num_target_tokens = loader.dataset.num_target_tokens
    # Switch dataset and model to "eval" mode
    loader.dataset.eval()
    model.eval()
    total_acc = AverageMeter()
    cot_acc = AverageMeter()
    ans_acc = AverageMeter()
    tokens_corr = {i: AverageMeter() for i in range(num_target_tokens)}
    bar = tqdm(loader)
    #model.set_cache(loader.dataset.device)
    for x in bar:
        y = x[:, num_prefix_tokens:].clone()
        x = x[:, :num_prefix_tokens].clone()
        y_pred = []
        for i in range(pass_at_k):
            with ctx:
                y_pred.append(
                    model.generate(x, max_new_tokens=num_target_tokens, 
                                #    temperature=temperature, 
                                #    top_k=top_k,
                                   do_sample=False, attention_mask = torch.ones_like(x), pad_token_id=2))
        y_pred = torch.stack(y_pred, dim=0)
        #model.reset_cache()
        # Check how many tokens we get right and how many predictions are completely correct
        correct = torch.stack([y.eq(y_pred[i, :, -num_target_tokens:]).float() for i in range(len(y_pred))])
        # Completely correct
        completely_correct = torch.tensor([
            torch.mean(correct[i].sum(dim=1).eq(num_target_tokens).to(torch.float)) for i in range(len(y_pred))]).to(y.device)
        total_acc.update(torch.max(completely_correct).item(), x.shape[0])
        
        cot_correct = torch.tensor([
            torch.mean(correct[i, :num_target_tokens//2].mean(dim=1).to(torch.float)) for i in range(len(y_pred))]).to(y.device)
        cot_acc.update(torch.max(cot_correct).item(), x.shape[0])
        
        ans_correct = torch.tensor([
            torch.mean(correct[i, num_target_tokens//2:].mean(dim=1).to(torch.float)) for i in range(len(y_pred))]).to(y.device)
        ans_acc.update(torch.max(ans_correct).item(), x.shape[0])
        
        # Individual token accuracy
        per_token_acc = correct.mean(dim=1).max(dim=0)[0]
        for i in range(num_target_tokens):
            tokens_corr[i].update(per_token_acc[i].item(), x.shape[0])
        bar.set_description(f'{mode} pass_at_{pass_at_k} accuracy: {total_acc.get(percentage=True):.2f}')
    #model.empty_cache()
    # Switch back to train mode
    loader.dataset.train()
    model.train()
    if results is not None:
        results[mode + '/full_accuracy'] = total_acc.get(percentage=True)
        results[mode + '/cot_accuracy'] = cot_acc.get(percentage=True)
        results[mode + '/ans_accuracy'] = ans_acc.get(percentage=True)
        for i in range(num_target_tokens):
            results[mode + '/token_' + str(i + 1)] = tokens_corr[i].get(percentage=True)
    return results



# Parse arguments
parser = argparse.ArgumentParser(description="Next-token failures")
# Data
parser.add_argument(
    "--n_samples", type=int, default=5, help="Number of samples to generate"
    )
parser.add_argument(
    "--model", default='gpt2', type=str, help="Type of model"
    )
parser.add_argument(
    "--dataset", default='graph', type=str, help="Choice of dataset"
    )
parser.add_argument(
    "--n_train", default=200000, type=int, help="Number of training samples"
    )
parser.add_argument(
    "--n_test", default=5000, type=int, help="Number of test samples"
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
        "--mate_in", default=2, type=int, help="For chess, number of moves to checkmate"
    )
parser.add_argument(
        "--unrolled", action=argparse.BooleanOptionalAction, default=True, help="For chess, unrolled board state",
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
        "--epochs_sft", type=int, default=1, help="Number of SFT epochs",
    )
parser.add_argument(
        "--save_every", type=int, default=5000, help="Interval (in steps) at which to save model",
    )
parser.add_argument(
        "--pass_at_k", type=int, default=1, help="pass at k eval",
    )
parser.add_argument(
        "--teacherless", action=argparse.BooleanOptionalAction, default=False, help="Standard or teacherless training",
    )
parser.add_argument(
        "--reverse", action=argparse.BooleanOptionalAction, default=False, help="Standard format or reverse targets",
    )
parser.add_argument(
        "--cot", action=argparse.BooleanOptionalAction, default=False, help="Standard format or cot targets",
    )
parser.add_argument(
        "--pos", action=argparse.BooleanOptionalAction, default=False, help="Standard format or pos tokens",
    )
parser.add_argument(
        "--eval_train", action=argparse.BooleanOptionalAction, default=False, help="Eval for training set",
    )
parser.add_argument(
        "--eval_every", type=int, default=400, help="Interval (in steps) to evaluate the model on test",
    )
parser.add_argument(
        "--use_wandb", action=argparse.BooleanOptionalAction, default=False, help="Whether to use wandb",
    )
parser.add_argument(
        "--wandb_entity", type=str, default=5000, help="Wandb username",
    )


args = parser.parse_args()
# System stuff
device = 'cuda' if torch.cuda.is_available() else 'cpu'
wandb_entity = args.wandb_entity
wandb_log = args.use_wandb
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Model stuff
top_k = 1000
temperature = 1.
pass_at_k = args.pass_at_k
n_samples = args.n_samples

# Evaluation stuff
eval_iters = 1000
eval_interval = 5
log_interval = 10

# Optimiser
dtype = 'bfloat16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
beta1 = 0.9
beta2 = 0.999
decay_lr = True
args.compile = False if device == 'cuda' else False
args.use_flash = True if device == 'cuda' else False
warmup_iters = 100
min_lr = 1e-5

run_name = get_run_name(args)
path = './checkpoints/' + run_name + '.pt'

# Get tokenizer and de-tokenizer
tokenizer = get_tokenizer(args)
# tokenizer = AutoTokenizer.from_pretrained('gpt2')
# tokenizer.pad_token = tokenizer.eos_token
train_data, test_data = get_dataset(args, tokenizer, device)

train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True)

target_len = train_data.num_tokens - train_data.num_prefix_tokens
max_iters = len(train_data) * args.epochs_sft
lr_decay_iters = max_iters

block_size = train_data.num_tokens
args.block_size = train_data.num_tokens
args.vocab_size = tokenizer.vocab_size
args.teacherless_token = tokenizer.encode('$')[0] if args.teacherless else None

# model = get_model(args)
from transformers import GPT2LMHeadModel
model = GPT2LMHeadModel.from_pretrained(args.model)

if args.compile:
    print("compiling the model... (takes a ~minute)")
    model = torch.compile(model)

model.to(device)
model.train()

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
ctx = nullcontext() if device == 'cpu' else torch.amp.autocast(device_type=device, dtype=ptdtype)

# Setup wandb logging
if wandb_log:
    run = wandb.init(project='next-token-failures', entity=wandb_entity, config=args.__dict__,)
    wandb.run.name = run_name

results = {}
num_iters = 0

for ep in range(args.epochs_sft):
    train_bar = tqdm(train_loader)
    total_loss, total_acc = AverageMeter(), AverageMeter()
    for x, y in train_bar:
        if num_iters % args.save_every == 0 and num_iters > 0:
            torch.save(
                model.state_dict(),
                path + "_epoch_" + str(ep)
            )
        # determine and set the learning rate for this iteration
        lr = get_lr(num_iters, args.lr, warmup_iters, lr_decay_iters, min_lr) if decay_lr else args.lr
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        with ctx:
            # logits, loss, accs = model(x, y)
            logits = model(x)['logits']
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
            acc = torch.mean((torch.argmax(logits[:, -train_data.num_target_tokens, :], dim=-1) == y[:, -train_data.num_target_tokens]).float())
        total_loss.update(loss.item(), x.shape[0] * train_data.num_target_tokens)
        total_acc.update(acc.item(), x.shape[0] * train_data.num_target_tokens)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        num_iters += 1
        train_bar.set_description(
            'Epoch: [{}/{}] Loss: {:.4f} Acc: {:.2f}'.format(ep, args.epochs_sft, total_loss.get(),
             total_acc.get(percentage=True))
        )
        # evaluate the loss on train/val sets and write checkpoints
        if num_iters % args.eval_every == 0 and num_iters > 1:
            # Generate sequences and check accuracies
            if args.eval_train:
                results = evaluate(model, train_loader, temperature=temperature, pass_at_k=pass_at_k, top_k=top_k, results=results, mode='Train')
                # results = evaluate_forced(model, train_loader, results=results, mode='train')
            results = evaluate(model, test_loader, temperature=temperature, pass_at_k=pass_at_k, ctx=ctx, top_k=top_k, results=results, mode='Test')
            # results = evaluate_forced(model, test_loader, ctx=ctx, results=results, mode='Test')
            pprint(results)
            if wandb_log:
                run.log(results)

results = evaluate(model, test_loader, temperature=temperature, pass_at_k=pass_at_k, ctx=ctx, top_k=top_k, results=results, mode='Test')
# results = evaluate_forced(model, test_loader, ctx=ctx, results=results, mode='Test')
# pprint(results)
if wandb_log:
    run.log(results)

# sample model generation and rate them

def create_dpo_dataset(prompts, generations, scores):
    n = scores.shape[0]
    chosen = []
    rejected = []
    final_prompts = []
    bar = tqdm(range(n))
    for i in bar:
        highest_score_id = torch.argmax(scores[i])
        lowest_score_id = torch.argmin(scores[i])
        # if scores[i][highest_score_id] == 1.0:
        chosen.append(generations[i, highest_score_id])
        rejected.append(generations[i, lowest_score_id])    
        final_prompts.append(prompts[i])
        bar.set_description(f'Chosen: {scores[i][highest_score_id]} Rejected: {scores[i][lowest_score_id]}')
    return Dataset.from_dict({'chosen': chosen, 'rejected': rejected, 'prompt': final_prompts})

# ref_model = deepcopy(model)
    
    
for i in range(10):
    ref_model = deepcopy(model)
    prompts, generations, scores = generate_and_score(model, train_loader, ctx, temperature=2.5, top_k=top_k, n_samples=n_samples)
    dpo_dataset = create_dpo_dataset(prompts, generations, scores)
    training_args = TrainingArguments(output_dir="./output", remove_unused_columns=False)
    training_args = training_args.set_training(learning_rate=args.lr, batch_size=args.batch_size)    
    dpo_tokenizer = AutoTokenizer.from_pretrained(args.model)
    dpo_tokenizer.pad_token_id = dpo_tokenizer.eos_token_id
    dpo_trainer = DPOTrainer(
        model, 
        ref_model=ref_model, 
        args=training_args, 
        train_dataset=dpo_dataset, 
        tokenizer=dpo_tokenizer,
        max_length=64,
        max_prompt_length=64,
        beta=1.)
    dpo_trainer.train()
    results = evaluate(model, test_loader, temperature=temperature, pass_at_k=pass_at_k, ctx=ctx, top_k=top_k, results=results, mode='Test')
    if wandb_log:
        run.log(results)

    # results = evaluate_forced(model, test_loader, ctx=ctx, results=results, mode='Test')
    # pprint(results)

# python -i online_dpo.py --cot --n_train 400 --n_test 500 --epochs_sft 50 --n_samples 50 --use_wandb --wandb_entity "ars22"
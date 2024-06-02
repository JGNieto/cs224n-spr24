'''
Multitask BERT class, starter training code, evaluation, and test code.

Of note are:
* class MultitaskBERT: Your implementation of multitask BERT.
* function train_multitask: Training procedure for MultitaskBERT. Starter code
    copies training procedure from `classifier.py` (single-task SST).
* function test_multitask: Test procedure for MultitaskBERT. This function generates
    the required files for submission.

Running `python multitask_classifier.py` trains and tests your MultitaskBERT and
writes all required submission files.
'''

import random, numpy as np, argparse
from types import SimpleNamespace

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from itertools import islice

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm
# from smart_pytorch import SMARTLoss

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask

from pcgrad import PCGrad
from dora import replace_linear_with_dora, replace_linear_with_simple_lora
from lora import replace_linear_with_lora

from datetime import datetime
import time

import gc
import os

TQDM_DISABLE=False


# Fix the random seed.
def seed_everything(seed=11711):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


BERT_HIDDEN_SIZE = 768

SENTIMENT_BATCH_SIZE = 8
PARAPHRASE_BATCH_SIZE = 8
STS_BATCH_SIZE = 8

MAX_PER_ITER = 100

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class MultitaskBERT(nn.Module):
    '''
    This module should use BERT for 3 tasks:

    - Sentiment classification (predict_sentiment)
    - Paraphrase detection (predict_paraphrase)
    - Semantic Textual Similarity (predict_similarity)
    '''
    def __init__(self, config):
        super(MultitaskBERT, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # last-linear-layer mode does not require updating BERT paramters.
        self.set_fine_tune_mode(config.fine_tune_mode)

        # You will want to add layers here to perform the downstream tasks.

        # Sentiment classification layers
        # self.sentiment_fc1 = nn.Linear(config.hidden_size, config.hidden_size // 2)
        # self.sentiment_fc2 = nn.Linear(config.hidden_size // 2, 5)
        # self.sentiment_activation = nn.ReLU()
        self.sentiment_dropout = nn.Dropout(config.last_dropout_prob)
        self.sentiment_linear = nn.Linear(config.hidden_size, 5)

        # Paraphrase detection layers
        # self.paraphrase_fc1 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        # self.paraphrase_fc2 = nn.Linear(config.hidden_size, 1)
        # self.paraphrase_activation = nn.ReLU()
        self.paraphrase_linear = nn.Linear(config.hidden_size, 1)
        self.paraphrase_dropout = nn.Dropout(config.last_dropout_prob)

        # Semantic textual similarity layers
        # self.similarity_fc1 = nn.Linear(2 * config.hidden_size, config.hidden_size)
        # self.similarity_fc2 = nn.Linear(config.hidden_size, 1)
        # self.similarity_activation = nn.ReLU()
        self.similarity_linear = nn.Linear(config.hidden_size, 1)
        self.similarity_dropout = nn.Dropout(config.last_dropout_prob)

    def set_fine_tune_mode(self, mode):
        assert mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if mode == 'last-linear-layer':
                param.requires_grad = False
            elif mode == 'full-model':
                param.requires_grad = True

    def forward(self, input_ids, attention_mask):
        'Takes a batch of sentences and produces embeddings for them.'
        # The final BERT embedding is the hidden state of [CLS] token (the first token)
        # Here, you can start by just returning the embeddings straight from BERT.
        # When thinking of improvements, you can later try modifying this
        # (e.g., by adding other layers).

        bert_outputs = self.bert(input_ids, attention_mask)
        return bert_outputs["pooler_output"]


    def predict_sentiment(self, input_ids, attention_mask):
        '''Given a batch of sentences, outputs logits for classifying sentiment.
        There are 5 sentiment classes:
        (0 - negative, 1- somewhat negative, 2- neutral, 3- somewhat positive, 4- positive)
        Thus, your output should contain 5 logits for each sentence.
        '''

        embeddings = self.forward(input_ids, attention_mask)
        # x = self.sentiment_dropout(embeddings)
        # x = self.sentiment_fc1(x)
        # x = self.sentiment_activation(x)
        # x = self.sentiment_dropout(x)
        # logits = self.sentiment_fc2(x)

        x = self.sentiment_dropout(embeddings)
        logits = self.sentiment_linear(x)

        return logits

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''

        embeddings_1 = self.forward(input_ids_1, attention_mask_1)
        # embeddings_2 = self.forward(input_ids_2, attention_mask_2)
        # combined_embeddings = torch.cat((embeddings_1, embeddings_2), 1)
        # x = self.paraphrase_dropout(combined_embeddings)
        # x = self.paraphrase_fc1(x)
        # x = self.paraphrase_activation(x)
        # x = self.paraphrase_dropout(x)
        # logit = self.paraphrase_fc2(x)

        x = self.paraphrase_dropout(embeddings_1)
        logit = self.paraphrase_linear(x)

        return logit


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''

        embeddings_1 = self.forward(input_ids_1, attention_mask_1)
        # embeddings_2 = self.forward(input_ids_2, attention_mask_2)
        # combined_embeddings = torch.cat((embeddings_1, embeddings_2), 1)
        # x = self.similarity_dropout(combined_embeddings)
        # x = self.similarity_fc1(x)
        # x = self.similarity_activation(x)
        # x = self.similarity_dropout(x)
        # logit = self.similarity_fc2(x)

        x = self.similarity_dropout(embeddings_1)
        logit = self.similarity_linear(x)

        return logit
    
    def compute_l1_loss(self, w):
        return torch.abs(w).sum()
  
    def compute_l2_loss(self, w):
        return torch.square(w).sum()


def save_model(model, optimizer, args, config, filepath):
    if isinstance(optimizer, PCGrad):
        optimizer = optimizer.optimizer

    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    # if args.lora:
    #     save_info['model'] = lora.lora_state_dict(model)
    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

def log(string, args):
    with open(args.stats, "a+") as f:
        f.write(string + "\n")
        print(string)

def compute_parameters(model, args):
    total_params = sum(p.numel() for p in model.parameters())
    log(f"Total parameters: {total_params}", args)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Trainable parameters: {trainable_params}", args)

# Custom function
def sentiment_batch(model: nn.Module, batch) -> torch.Tensor:
    b_ids, b_mask, b_labels = (batch['token_ids'],
                               batch['attention_mask'], batch['labels'])

    b_ids = b_ids.to(DEVICE)
    b_mask = b_mask.to(DEVICE)
    b_labels = b_labels.to(DEVICE)

    logits = model.predict_sentiment(b_ids, b_mask)
    loss = F.cross_entropy(logits, b_labels.view(-1))

    return loss

# Custom function
def paraphrase_batch(model: nn.Module, batch) -> torch.Tensor:
    b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                                      batch['attention_mask_1'],
                                                      batch['token_ids_2'],
                                                      batch['attention_mask_2'],
                                                      batch['labels'])

    b_ids_1 = b_ids_1.to(DEVICE)
    b_mask_1 = b_mask_1.to(DEVICE)
    b_ids_2 = b_ids_2.to(DEVICE)
    b_mask_2 = b_mask_2.to(DEVICE)
    b_labels = b_labels.to(DEVICE)

    logit = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)

    loss = F.binary_cross_entropy_with_logits(logit.view(-1), b_labels.float())

    return loss

# Custom function
def semantic_batch(model: nn.Module, batch) -> torch.Tensor:
    b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                                      batch['attention_mask_1'],
                                                      batch['token_ids_2'],
                                                      batch['attention_mask_2'],
                                                      batch['labels'])

    b_ids_1 = b_ids_1.to(DEVICE)
    b_mask_1 = b_mask_1.to(DEVICE)
    b_ids_2 = b_ids_2.to(DEVICE)
    b_mask_2 = b_mask_2.to(DEVICE)
    b_labels = b_labels.to(DEVICE)


    logit = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
    loss = F.mse_loss(logit.view(-1), b_labels.float())
    # loss = F.smooth_l1_loss(logit.view(-1), b_labels.float())

    return loss

def overall_score(sst_acc, para_acc, sts_corr):
    scores = []
    if sst_acc is not None:
        scores.append(sst_acc)
    if para_acc is not None:
        scores.append(para_acc)
    if sts_corr is not None:
        scores.append((sts_corr + 1) / 2)

    return np.mean(scores)

def train_single_task(args):
    train_data = None
    dev_data = None

    train_dataloader = None
    sst_dev_dataloader = None
    para_dev_dataloader = None
    sts_dev_dataloader = None

    function = None

    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    if args.task == 'sst':
        train_data = SentenceClassificationDataset(sst_train_data, args)
        dev_data = SentenceClassificationDataset(sst_dev_data, args)

        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=SENTIMENT_BATCH_SIZE,
                                          collate_fn=train_data.collate_fn)
        sst_dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=SENTIMENT_BATCH_SIZE,
                                        collate_fn=dev_data.collate_fn)
        function = sentiment_batch

    elif args.task == 'para':
        train_data = SentencePairDataset(para_train_data, args)
        dev_data = SentencePairDataset(para_dev_data, args)

        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=PARAPHRASE_BATCH_SIZE,
                                          collate_fn=train_data.collate_fn)
        para_dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=PARAPHRASE_BATCH_SIZE,
                                         collate_fn=dev_data.collate_fn)
        function = paraphrase_batch

    elif args.task == 'sts':
        train_data = SentencePairDataset(sts_train_data, args)
        dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        train_dataloader = DataLoader(train_data, shuffle=True, batch_size=STS_BATCH_SIZE,
                                         collate_fn=train_data.collate_fn)
        sts_dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=STS_BATCH_SIZE,
                                            collate_fn=dev_data.collate_fn)
        function = semantic_batch

    else:
        print("Invalid task")
        exit(1)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'last_dropout_prob': args.last_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': BERT_HIDDEN_SIZE,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    if args.parallel:
        model = torch.nn.DataParallel(model)
    else:
        model = model.to(DEVICE)

    if args.load:
        saved = torch.load(args.load)
        if args.dora:
            replace_linear_with_dora(model, DEVICE)
        elif args.lora:
            replace_linear_with_simple_lora(model, DEVICE)
        model.load_state_dict(saved['model'])
        config = saved['model_config']
        log(f"Loaded model from {args.load}", args)
    else:
        if args.dora:
            log("Using DoRA", args)
            replace_linear_with_dora(model, DEVICE)
        elif args.lora:
            log("Using LoRA", args)
            replace_linear_with_simple_lora(model, DEVICE)
        else:
            log("Not using DoRA or LoRA", args)

    lr = args.lr
    optimizer: AdamW = AdamW(model.parameters(), lr=lr, weight_decay=args.decay)
    best_dev_acc = 0
    n_discarded = 0

    log("Using AdamW", args)
    

    log("Start training at time: " + str(datetime.now()), args)

    last_good_epoch = 0
    epoch_times = []
    training_start = time.time()

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        print()
        print("Collecting garbage...")
        gc.collect()
        if DEVICE.type == 'cuda':
            print("Emptying CUDA cache...")
            torch.cuda.empty_cache()

        model.train()
        train_loss = 0
        num_batches = 0

        start = time.time()

        for batch in tqdm(train_dataloader, desc=f'train-{epoch}', disable=TQDM_DISABLE):
            loss = function(model, batch)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        elapsed = time.time() - start
        epoch_times.append(elapsed)
        log(f"Epoch {epoch} took {elapsed:.2f} seconds", args)

        train_loss = train_loss / num_batches

        # sst_train_acc, _, _, para_train_acc, _, _, sts_train_corr, *_  = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, DEVICE)
        sst_dev_acc, _, _, para_dev_acc, _, _, sts_dev_corr, *_ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, DEVICE)

        dev_acc = overall_score(sst_dev_acc, para_dev_acc, sts_dev_corr)

        log(f"SST: {sst_dev_acc}, Para: {para_dev_acc}, STS: {sts_dev_corr}", args)
        log(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}", args)

        if dev_acc > best_dev_acc:
            log(f"New best dev acc :: {dev_acc :.3f} (prev: {best_dev_acc :.3f})", args)
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)
            n_discarded = 0
            last_good_epoch = epoch
        else:
            log(f"Discard model (best dev acc :: {best_dev_acc :.3f})", args)
            n_discarded += 1
            if n_discarded >= args.early_stop:
                log(f"Early stopping after {n_discarded} discarded models", args)
                break


    log("Last good epoch: " + str(last_good_epoch), args)
    log("Finish training at time: " + str(datetime.now()), args)
    log("Total training time: " + str(time.time() - training_start) + " seconds.", args)
    log("Average epoch time: " + str(np.mean(epoch_times)) + " seconds.", args)

def ratios(arr):
    min_val = min(arr)
    return (min(x // min_val, MAX_PER_ITER) for x in arr)

def execute_batch(model: nn.Module, iter, function, n) -> torch.Tensor:
    loss: torch.Tensor | None = None
    for _ in range(n):
        batch = next(iter)
        dloss = function(model, batch)
        if loss is None:
            loss = dloss
        else:
            loss += dloss

    assert loss is not None

    return loss / n

def train_multitask(args):
    '''Train MultitaskBERT.
    '''
    # Create the data and its corresponding datasets and dataloader.
    sst_train_data, num_labels,para_train_data, sts_train_data = load_multitask_data(args.sst_train,args.para_train,args.sts_train, split ='train')
    sst_dev_data, num_labels,para_dev_data, sts_dev_data = load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev, split ='train')

    sst_train_data = SentenceClassificationDataset(sst_train_data, args)
    sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

    sst_train_dataloader = DataLoader(sst_train_data, shuffle=True, batch_size=SENTIMENT_BATCH_SIZE,
                                      collate_fn=sst_train_data.collate_fn)
    sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=SENTIMENT_BATCH_SIZE,
                                    collate_fn=sst_dev_data.collate_fn)

    para_train_data = SentencePairDataset(para_train_data, args)
    para_dev_data = SentencePairDataset(para_dev_data, args)

    para_train_dataloader = DataLoader(para_train_data, shuffle=True, batch_size=PARAPHRASE_BATCH_SIZE,
                                      collate_fn=para_train_data.collate_fn)
    para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=PARAPHRASE_BATCH_SIZE,
                                     collate_fn=para_dev_data.collate_fn)

    sts_train_data = SentencePairDataset(sts_train_data, args)
    sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

    sts_train_dataloader = DataLoader(sts_train_data, shuffle=True, batch_size=STS_BATCH_SIZE,
                                     collate_fn=sts_train_data.collate_fn)
    sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=STS_BATCH_SIZE,
                                        collate_fn=sts_dev_data.collate_fn)

    # Init model.
    config = {'hidden_dropout_prob': args.hidden_dropout_prob,
              'last_dropout_prob': args.last_dropout_prob,
              'num_labels': num_labels,
              'hidden_size': BERT_HIDDEN_SIZE,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    if args.parallel:
        model = torch.nn.DataParallel(model)
    else:
        model = model.to(DEVICE)

    if args.load:
        saved = torch.load(args.load, device=DEVICE)
        if args.dora:
            replace_linear_with_dora(model, DEVICE)
        elif args.lora:
            replace_linear_with_simple_lora(model, DEVICE)
        model.load_state_dict(saved['model'])
        config = saved['model_config']
        log(f"Loaded model from {args.load}", args)
    else:
        if args.dora:
            log("Using DoRA", args)
            replace_linear_with_dora(model, DEVICE)
        elif args.lora:
            log("Using LoRA", args)
            replace_linear_with_simple_lora(model, DEVICE)
        else:
            log("Not using DoRA or LoRA", args)

    lr = args.lr
    optimizer: AdamW | PCGrad = AdamW(model.parameters(), lr=lr, weight_decay=args.decay)
    best_dev_acc = 0
    n_discarded = 0

    sst_len = len(sst_train_data) // SENTIMENT_BATCH_SIZE
    para_len = len(para_train_data) // PARAPHRASE_BATCH_SIZE
    sts_len = len(sts_train_data) // STS_BATCH_SIZE

    min_num_samples = min(sst_len, para_len, sts_len)
    n_sst, n_para, n_sts = ratios([sst_len, para_len, sts_len])

    if args.one_at_a_time:
        n_sst, n_para, n_sts = 1, 1, 1
        log("One at a time", args)
    else:
        log(f"SST: {n_sst}, Para: {n_para}, STS: {n_sts}", args)

    compute_parameters(model, args)

    log(f"Samples each iteration... SST: {n_sst}, Para: {n_para}, STS: {n_sts}", args)
    if args.pcgrad:
        log("Using PCGrad", args)
        optimizer = PCGrad(optimizer)
    else:
        log("Using AdamW", args)

    log("Start training at time: " + str(datetime.now()), args)

    last_good_epoch = 0
    epoch_times = []
    training_start = time.time()

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        print("Collecting garbage...")
        gc.collect()
        if DEVICE.type == 'cuda':
            print("Emptying CUDA cache...")
            torch.cuda.empty_cache()

        model.train()
        train_loss = 0
        num_batches = 0

        sst_iter = iter(sst_train_dataloader)
        para_iter = iter(para_train_dataloader)
        sts_iter = iter(sts_train_dataloader)

        start = time.time()

        for _ in tqdm(range(min_num_samples), desc=f'train-{epoch}', disable=TQDM_DISABLE):
            try:
                sentiment_loss = execute_batch(model, sst_iter, sentiment_batch, n_sst)
                paraphrase_loss = execute_batch(model, para_iter, paraphrase_batch, n_para)
                semantic_loss = execute_batch(model, sts_iter, semantic_batch, n_sts)

                losses = [sentiment_loss, paraphrase_loss, semantic_loss]
                loss: torch.Tensor = sentiment_loss + paraphrase_loss + semantic_loss

                if type(loss) != torch.Tensor:
                    print("Loss is not a tensor")
                    print(loss)
                    print(type(loss))
                    print(losses)
                    print(sentiment_loss)
                    print(paraphrase_loss)
                    print(semantic_loss)
                    continue

                if args.l1l2:
                    # Specify L1 and L2 weights
                    l1_weight = 0.3
                    l2_weight = 0.7
                    # Compute L1 and L2 loss component
                    parameters = []
                    for parameter in model.parameters():
                        parameters.append(parameter.view(-1))
                    l1 = l1_weight * model.compute_l1_loss(torch.cat(parameters))
                    l2 = l2_weight * model.compute_l2_loss(torch.cat(parameters))
                    
                    # Regularization: Add L1 and L2 loss components
                    loss += l1
                    loss += l2

                optimizer.zero_grad()

                if isinstance(optimizer, PCGrad):
                    optimizer.pc_backward(losses) # type: ignore
                else:
                    loss.backward()

                optimizer.step()

                train_loss += loss.item()
                num_batches += 1
            except StopIteration:
                print("StopIteration!!")
                break

        elapsed = time.time() - start
        epoch_times.append(elapsed)
        log(f"Epoch {epoch} took {elapsed:.2f} seconds", args)

        train_loss = train_loss / num_batches

        # sst_train_acc, _, _, para_train_acc, _, _, sts_train_corr, *_  = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, DEVICE)
        sst_dev_acc, _, _, para_dev_acc, _, _, sts_dev_corr, *_ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, DEVICE)

        dev_acc = overall_score(sst_dev_acc, para_dev_acc, sts_dev_corr)

        log(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}", args)

        if dev_acc > best_dev_acc:
            log(f"New best dev acc :: {dev_acc :.3f} (prev: {best_dev_acc :.3f})", args)
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)
            n_discarded = 0
            last_good_epoch = epoch
        else:
            log(f"Discard model (best dev acc :: {best_dev_acc :.3f})", args)
            n_discarded += 1
            if n_discarded >= args.early_stop:
                log(f"Early stopping after {n_discarded} discarded models", args)
                break


    log("Last good epoch: " + str(last_good_epoch), args)
    log("Finish training at time: " + str(datetime.now()), args)
    log("Total training time: " + str(time.time() - training_start) + " seconds.", args)
    log("Average epoch time: " + str(np.mean(epoch_times)) + " seconds.", args)

def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        if args.parallel:
            model = torch.nn.DataParallel(model)
        else:
            model = model.to(DEVICE)
        if args.dora:
            replace_linear_with_dora(model, DEVICE)
        elif args.lora:
            replace_linear_with_simple_lora(model, DEVICE)
        model.load_state_dict(saved['model'])
        print(f"Loaded model to test from {args.filepath}")

        sst_test_data, num_labels,para_test_data, sts_test_data = \
            load_multitask_data(args.sst_test,args.para_test, args.sts_test, split='test')

        sst_dev_data, num_labels,para_dev_data, sts_dev_data = \
            load_multitask_data(args.sst_dev,args.para_dev,args.sts_dev,split='dev')

        sst_test_data = SentenceClassificationTestDataset(sst_test_data, args)
        sst_dev_data = SentenceClassificationDataset(sst_dev_data, args)

        sst_test_dataloader = DataLoader(sst_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sst_test_data.collate_fn)
        sst_dev_dataloader = DataLoader(sst_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sst_dev_data.collate_fn)

        para_test_data = SentencePairTestDataset(para_test_data, args)
        para_dev_data = SentencePairDataset(para_dev_data, args)

        para_test_dataloader = DataLoader(para_test_data, shuffle=True, batch_size=args.batch_size,
                                          collate_fn=para_test_data.collate_fn)
        para_dev_dataloader = DataLoader(para_dev_data, shuffle=False, batch_size=args.batch_size,
                                         collate_fn=para_dev_data.collate_fn)

        sts_test_data = SentencePairTestDataset(sts_test_data, args)
        sts_dev_data = SentencePairDataset(sts_dev_data, args, isRegression=True)

        sts_test_dataloader = DataLoader(sts_test_data, shuffle=True, batch_size=args.batch_size,
                                         collate_fn=sts_test_data.collate_fn)
        sts_dev_dataloader = DataLoader(sts_dev_data, shuffle=False, batch_size=args.batch_size,
                                        collate_fn=sts_dev_data.collate_fn)

        dev_sentiment_accuracy,dev_sst_y_pred, dev_sst_sent_ids, \
            dev_paraphrase_accuracy, dev_para_y_pred, dev_para_sent_ids, \
            dev_sts_corr, dev_sts_y_pred, dev_sts_sent_ids = model_eval_multitask(sst_dev_dataloader,
                                                                    para_dev_dataloader,
                                                                    sts_dev_dataloader, model, DEVICE)

        test_sst_y_pred, \
            test_sst_sent_ids, test_para_y_pred, test_para_sent_ids, test_sts_y_pred, test_sts_sent_ids = \
                model_eval_test_multitask(sst_test_dataloader,
                                          para_test_dataloader,
                                          sts_test_dataloader, model, DEVICE)

        with open(args.sst_dev_out, "w+") as f:
            log(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}", args)
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            log(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}", args)
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            log(f"dev sts corr :: {dev_sts_corr :.3f}", args)
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(dev_sts_sent_ids, dev_sts_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_test_out, "w+") as f:
            f.write(f"id \t Predicted_Similiary \n")
            for p, s in zip(test_sts_sent_ids, test_sts_y_pred):
                f.write(f"{p} , {s} \n")


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sst_train", type=str, default="data/ids-sst-train.csv")
    parser.add_argument("--sst_dev", type=str, default="data/ids-sst-dev.csv")
    parser.add_argument("--sst_test", type=str, default="data/ids-sst-test-student.csv")

    parser.add_argument("--para_train", type=str, default="data/quora-train.csv")
    parser.add_argument("--para_dev", type=str, default="data/quora-dev.csv")
    parser.add_argument("--para_test", type=str, default="data/quora-test-student.csv")

    parser.add_argument("--sts_train", type=str, default="data/sts-train.csv")
    parser.add_argument("--sts_dev", type=str, default="data/sts-dev.csv")
    parser.add_argument("--sts_test", type=str, default="data/sts-test-student.csv")

    parser.add_argument("--seed", type=int, default=11711)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine-tune-mode", type=str,
                        help='last-linear-layer: the BERT parameters are frozen and the task specific head parameters are updated; full-model: BERT parameters are updated as well',
                        choices=('last-linear-layer', 'full-model'), default="last-linear-layer")

    parser.add_argument("--sst_dev_out", type=str, default="predictions/sst-dev-output.csv")
    parser.add_argument("--sst_test_out", type=str, default="predictions/sst-test-output.csv")

    parser.add_argument("--para_dev_out", type=str, default="predictions/para-dev-output.csv")
    parser.add_argument("--para_test_out", type=str, default="predictions/para-test-output.csv")

    parser.add_argument("--sts_dev_out", type=str, default="predictions/sts-dev-output.csv")
    parser.add_argument("--sts_test_out", type=str, default="predictions/sts-test-output.csv")

    parser.add_argument("--batch_size", help='ONLY TEST sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--last_dropout_prob", type=float, default=0.4)
    parser.add_argument("--lr", type=float, help="learning rate", default=5e-5)

    parser.add_argument("--pcgrad", action='store_true', help='Use PCGrad instead of plain AdamW (only for multitask training)')
    parser.add_argument("--dora", action='store_true', help='Use DoRA PEFT')
    parser.add_argument("--lora", action='store_true', help='Use LoRA PEFT')
    parser.add_argument("--l1l2", action='store_true', help='Use L1 L2 Loss')
    parser.add_argument("--eval", action='store_true', help='Only evaluate the model, no training. Must use --load to specify .pt file')
    parser.add_argument("--parallel", action='store_true', help='Use parallel training')
    parser.add_argument("--task", type=str, help='sst for sentiment analysis, para for paraphrase detection, sts for semantic textual similarity and multi for multitask training all of them at once (by dafult, multitask training)', choices=('sst', 'para', 'sts', 'multi'), default='multi')
    parser.add_argument("--load", type=str, help='Load model from file')
    parser.add_argument("--decay", type=float, help='Weight decay', default=0.01)
    parser.add_argument("--early_stop", type=int, help='After this many models have been discarded, we stop. Default is no stop.', default=-1)
    parser.add_argument("--nickname", type=str, help='Nickname for the model', default='')
    parser.add_argument("--output", type=str, help='Output directory for model and logs', default='.')
    parser.add_argument("--one_at_a_time", action='store_true', help='Each training iteration, we will take one data point from each dataset, no matter their size.')

    args = parser.parse_args()
    return args

def common_logs(args):
    log(f"Fine-tune mode: {args.fine_tune_mode}", args)
    log(f"Learning rate: {args.lr}", args)
    log(f"Device: {DEVICE}", args)
    log(f"Task: {args.task}", args)
    log(f"Early stop: {args.early_stop}", args)
    log(f"Hidden droput: {args.hidden_dropout_prob}", args)
    log(f"Last dropout: {args.last_dropout_prob}", args)
    log(f"L1L2: {args.l1l2}", args)
    log(f"Decay: {args.decay}", args)

def run(args):
    assert not (args.dora and args.lora), "Cannot use both DoRA and LoRA at the same time LOL."
    assert not (args.dora or args.lora) or args.fine_tune_mode == 'full-model', "DoRA and LoRA can only be used with full-model fine-tuning."

    for x in [args.sst_dev_out, args.sst_test_out, args.para_dev_out, args.para_test_out, args.sts_dev_out, args.sts_test_out]:
        os.makedirs(os.path.dirname(x), exist_ok=True)

    args.output = os.path.dirname(args.output) 
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(os.path.join(args.output, "logs"), exist_ok=True)

    print("Nickname:", args.nickname)

    if args.early_stop == -1:
        args.early_stop = args.epochs + 1

    if args.eval:
        path = datetime.now().strftime('%Y-%m-%d-%H-%M') + ('' if args.nickname == '' else f"-{args.nickname}")
        if not args.load:
            print("You must specify a .pt file using --load to load the model from.")
            exit(1)
        args.filepath = args.load
        args.stats = os.path.join(args.output, f'test-{path}-stats.txt') # Stats path.
        test_multitask(args)
    else:
        path = datetime.now().strftime('%Y-%m-%d-%H-%M') + ('' if args.nickname == '' else f"-{args.nickname}") + f"-{args.fine_tune_mode}-{args.epochs}-{args.lr}-{'pcgrad' if args.pcgrad else 'adamw'}-{'dora' if args.dora else ('lora' if args.lora else 'swiper')}-{'l1l2' if args.l1l2 else 'regloss'}"

        seed_everything(args.seed)  # Fix the seed for reproducibility.

        if args.task == 'multi':
            args.filepath = os.path.join(args.output, f'{path}-multitask.pt') # Save path.
            args.stats = os.path.join(args.output, "logs", f'{path}-multitask-stats.txt') # Stats path.
            common_logs(args)
            train_multitask(args)
        else:
            args.filepath = os.path.join(args.output, f'{path}-{args.task}.pt') # Save path.
            args.stats = os.path.join(args.output, "logs", f'{path}-{args.task}-stats.txt') # Stats path.
            common_logs(args)
            train_single_task(args)

        test_multitask(args)

if __name__ == "__main__":
    args = get_args()
    run(args)


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
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler

from bert import BertModel
from optimizer import AdamW
from tqdm import tqdm

from datasets import (
    SentenceClassificationDataset,
    SentenceClassificationTestDataset,
    SentencePairDataset,
    SentencePairTestDataset,
    load_multitask_data
)

from evaluation import model_eval_sst, model_eval_multitask, model_eval_test_multitask

from pcgrad import PCGrad
from dora import replace_linear_with_dora

from datetime import datetime

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

N_SENTIMENT_CLASSES = 5     # 0 (negative) to 5 (positive)
N_STS_CLASSES = 6           # 0 (not at all related), to 5
N_PARAPHRASE_CLASSES = 2    # Binary classification

SENTIMENT_BATCH_SIZE = 8
STS_BATCH_SIZE = 8
PARAPHRASE_BATCH_SIZE = 8

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
        assert config.fine_tune_mode in ["last-linear-layer", "full-model"]
        for param in self.bert.parameters():
            if config.fine_tune_mode == 'last-linear-layer':
                param.requires_grad = False
            elif config.fine_tune_mode == 'full-model':
                param.requires_grad = True
        # You will want to add layers here to perform the downstream tasks.

        self.sentiment_linear = nn.Linear(config.hidden_size, N_SENTIMENT_CLASSES)
        self.sentiment_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.paraphrase_linear = nn.Linear(2 * config.hidden_size, N_PARAPHRASE_CLASSES)
        self.paraphrase_dropout = nn.Dropout(config.hidden_dropout_prob)

        self.similarity_linear = nn.Linear(2 * config.hidden_size, N_STS_CLASSES)
        self.similarity_dropout = nn.Dropout(config.hidden_dropout_prob)


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
        dropped = self.sentiment_dropout(embeddings)
        logits = self.sentiment_linear(dropped)
        return  logits

    def predict_paraphrase(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit for predicting whether they are paraphrases.
        Note that your output should be unnormalized (a logit); it will be passed to the sigmoid function
        during evaluation.
        '''

        embeddings_1 = self.forward(input_ids_1, attention_mask_1)
        embeddings_2 = self.forward(input_ids_2, attention_mask_2)


        combined_embeddings = torch.cat((embeddings_1, embeddings_2), 1)
        dropped = self.paraphrase_dropout(combined_embeddings)

        logits = self.paraphrase_linear(dropped)

        return  logits


    def predict_similarity(self,
                           input_ids_1, attention_mask_1,
                           input_ids_2, attention_mask_2):
        '''Given a batch of pairs of sentences, outputs a single logit corresponding to how similar they are.
        Note that your output should be unnormalized (a logit).
        '''

        embeddings_1 = self.forward(input_ids_1, attention_mask_1)
        embeddings_2 = self.forward(input_ids_2, attention_mask_2)

        combined_embeddings = torch.cat((embeddings_1, embeddings_2), 1)
        dropped = self.similarity_dropout(combined_embeddings)

        logits = self.similarity_linear(dropped)

        return  logits


def save_model(model, optimizer, args, config, filepath):
    save_info = {
        'model': model.state_dict(),
        'optim': optimizer.state_dict(),
        'args': args,
        'model_config': config,
        'system_rng': random.getstate(),
        'numpy_rng': np.random.get_state(),
        'torch_rng': torch.random.get_rng_state(),
    }

    torch.save(save_info, filepath)
    print(f"save the model to {filepath}")

# Custom function
def sentiment_batch(model: MultitaskBERT, batch):
    b_ids, b_mask, b_labels = (batch['token_ids'],
                               batch['attention_mask'], batch['labels'])

    batch_size = b_ids.size(0)

    assert batch_size == SENTIMENT_BATCH_SIZE

    b_ids = b_ids.to(DEVICE)
    b_mask = b_mask.to(DEVICE)
    b_labels = b_labels.to(DEVICE)

    logits = model.predict_sentiment(b_ids, b_mask)
    loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / batch_size

    return loss

# Custom function
def paraphrase_batch(model: MultitaskBERT, batch):
    b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                                      batch['attention_mask_1'],
                                                      batch['token_ids_2'],
                                                      batch['attention_mask_2'],
                                                      batch['labels'])

    batch_size = b_ids_1.size(0)

    assert batch_size == PARAPHRASE_BATCH_SIZE

    b_ids_1 = b_ids_1.to(DEVICE)
    b_mask_1 = b_mask_1.to(DEVICE)
    b_ids_2 = b_ids_2.to(DEVICE)
    b_mask_2 = b_mask_2.to(DEVICE)
    b_labels = b_labels.to(DEVICE)

    logits = model.predict_paraphrase(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
    loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / batch_size

    return loss

# Custom function
def semantic_batch(model: MultitaskBERT, batch):
    b_ids_1, b_mask_1, b_ids_2, b_mask_2, b_labels = (batch['token_ids_1'],
                                                      batch['attention_mask_1'],
                                                      batch['token_ids_2'],
                                                      batch['attention_mask_2'],
                                                      batch['labels'])

    batch_size = b_ids_1.size(0)

    assert batch_size == STS_BATCH_SIZE

    b_ids_1 = b_ids_1.to(DEVICE)
    b_mask_1 = b_mask_1.to(DEVICE)
    b_ids_2 = b_ids_2.to(DEVICE)
    b_mask_2 = b_mask_2.to(DEVICE)
    b_labels = b_labels.to(DEVICE)

    logits = model.predict_similarity(b_ids_1, b_mask_1, b_ids_2, b_mask_2)
    loss = F.cross_entropy(logits, b_labels.view(-1), reduction='sum') / batch_size

    return loss

def train_multitask(args):
    '''Train MultitaskBERT.

    Currently only trains on SST dataset. The way you incorporate training examples
    from other datasets into the training procedure is up to you. To begin, take a
    look at test_multitask below to see how you can use the custom torch `Dataset`s
    in datasets.py to load in examples from the Quora and SemEval datasets.
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
              'num_labels': num_labels,
              'hidden_size': BERT_HIDDEN_SIZE,
              'data_dir': '.',
              'fine_tune_mode': args.fine_tune_mode}

    config = SimpleNamespace(**config)

    model = MultitaskBERT(config)
    model = model.to(DEVICE)

    # UNCOMMENT this line to use DoRA
    # replace_linear_with_dora(model)

    lr = args.lr
    optimizer = AdamW(model.parameters(), lr=lr)
    best_dev_acc = 0

    num_samples = min(len(sst_train_data), len(para_train_data), len(sts_train_data))

    print("Number of samples:", num_samples)
    print("Start training at time:", datetime.now())

    # UNCOMMENT this line to use PCGrad
    # optimizer = PCGrad(tf.train.AdamW(model.parameters(), lr=lr))

    # Run for the specified number of epochs.
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        num_batches = 0

        for sst_batch, para_batch, sts_batch in tqdm(zip(sst_train_dataloader, para_train_dataloader, sts_train_dataloader), desc=f'train-{epoch}', disable=TQDM_DISABLE):
            sentiment_loss = sentiment_batch(model, sst_batch)
            paraphrase_loss = paraphrase_batch(model, para_batch)
            semantic_loss = semantic_batch(model, sts_batch)

            loss = sentiment_loss + paraphrase_loss + semantic_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        train_loss = train_loss / num_batches

        # sst_train_acc, _, _, para_train_acc, _, _, sts_train_corr, *_  = model_eval_multitask(sst_train_dataloader, para_train_dataloader, sts_train_dataloader, model, DEVICE)
        sst_dev_acc, _, _, para_dev_acc, _, _, sts_dev_corr, *_ = model_eval_multitask(sst_dev_dataloader, para_dev_dataloader, sts_dev_dataloader, model, DEVICE)

        sst_train_acc, para_train_acc, sts_train_corr = 0, 0, 0

        dev_acc = sst_dev_acc + para_dev_acc + sts_dev_corr

        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            save_model(model, optimizer, args, config, args.filepath)

        print(f"Epoch {epoch}: train loss :: {train_loss :.3f}, dev acc :: {dev_acc :.3f}")
        print(f"Sentiment train acc :: {sst_train_acc :.3f}, Sentiment dev acc :: {sst_dev_acc :.3f}")
        print(f"Paraphrase train acc :: {para_train_acc :.3f}, Paraphrase dev acc :: {para_dev_acc :.3f}")
        print(f"STS train corr :: {sts_train_corr :.3f}, STS dev corr :: {sts_dev_corr :.3f}")

    print("Finish training at time:", datetime.now())


def test_multitask(args):
    '''Test and save predictions on the dev and test sets of all three tasks.'''
    with torch.no_grad():
        saved = torch.load(args.filepath)
        config = saved['model_config']

        model = MultitaskBERT(config)
        model.load_state_dict(saved['model'])
        model = model.to(DEVICE)
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
            print(f"dev sentiment acc :: {dev_sentiment_accuracy :.3f}")
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(dev_sst_sent_ids, dev_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sst_test_out, "w+") as f:
            f.write(f"id \t Predicted_Sentiment \n")
            for p, s in zip(test_sst_sent_ids, test_sst_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_dev_out, "w+") as f:
            print(f"dev paraphrase acc :: {dev_paraphrase_accuracy :.3f}")
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(dev_para_sent_ids, dev_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.para_test_out, "w+") as f:
            f.write(f"id \t Predicted_Is_Paraphrase \n")
            for p, s in zip(test_para_sent_ids, test_para_y_pred):
                f.write(f"{p} , {s} \n")

        with open(args.sts_dev_out, "w+") as f:
            print(f"dev sts corr :: {dev_sts_corr :.3f}")
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

    parser.add_argument("--batch_size", help='sst: 64, cfimdb: 8 can fit a 12GB GPU', type=int, default=8)
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, help="learning rate", default=1e-5)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    args.filepath = f'{args.fine_tune_mode}-{args.epochs}-{args.lr}-multitask.pt' # Save path.
    seed_everything(args.seed)  # Fix the seed for reproducibility.
    train_multitask(args)
    test_multitask(args)

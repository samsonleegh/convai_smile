# no local GPU, trained on https://colab.research.google.com/drive/1Dlx9aOhQ8ODyhNRHqJBKDj2GSj5FRbB4?usp=sharing

import os
import argparse
import itertools
import operator
import tarfile
import tempfile
import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.metrics import f1_score
from simpletransformers.conv_ai import ConvAIModel, ConvAIArgs
from transformers import cached_path
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer

# create evaluation function to get F1 and perplexity score for lm as well


def evaluation(tune_model, eval_file):
    """ model evaluation on evaluation dataset
    : tune_model param: current model loaded from ConvAIModel class
    : eval_file param: filepath to evaluation dataset in json format
    : return results: dictionary of language model loss and F1 score, MC F1 score
    """
    tune_model._move_model_to_device()
    device = tune_model.device
    model = tune_model.model
    args = tune_model.args

    # set up recall and precision score for language model predictions
    eval_dataloader, eval_sampler = tune_model.load_and_cache_examples(
        dataset_path=eval_file,
        evaluate=True,
        verbose=False,
        silent=True,
        no_cache=True,
    )

    nb_eval_steps = 0
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
    results = {
        "language_model_loss": [],
        "lm_f1_score": [],
        "mc_f1_score": [],
    }
    # set to evaluation
    model.eval()
    for batch in eval_dataloader:
        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            input_ids, mc_token_ids, lm_labels, mc_labels, token_type_ids = batch

            lm_logits, mc_logits, * \
                _ = model(input_ids, token_type_ids=token_type_ids,
                          mc_token_ids=mc_token_ids,)
            # model outputs are always tuple in pytorch-transformers (see doc)

            lm_logits_flat_shifted = lm_logits[..., :-1,
                                               :].contiguous().view(-1, lm_logits.size(-1))
            lm_labels_flat_shifted = lm_labels[..., 1:].contiguous().view(-1)

        nb_eval_steps += 1

        mc_logits = [np.argmax(pred) for pred in mc_logits.cpu().numpy()]
        mc_f1_current = f1_score(
            mc_labels.cpu().numpy(), mc_logits, average="macro")
        lm_pred = [np.argmax(pred)
                   for pred in lm_logits_flat_shifted.cpu().numpy()]
        lm_f1_current = f1_score(
            lm_labels_flat_shifted.cpu().numpy(), lm_pred, average="macro")
        lm_loss_current = loss_fct(
            lm_logits_flat_shifted, lm_labels_flat_shifted)

        results["language_model_loss"].append(
            lm_loss_current.cpu().numpy().item())
        results["lm_f1_score"].append(lm_f1_current)
        results["mc_f1_score"].append(mc_f1_current)

    results["language_model_loss"] = statistics.mean(
        results["language_model_loss"])
    results["lm_f1_score"] = statistics.mean(results["lm_f1_score"])
    results["mc_f1_score"] = statistics.mean(results["mc_f1_score"])

    return results


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, default="data/counsel_before_sun_train.json",
                        help='Path to json training file')
    parser.add_argument('--eval_file', type=str, default="data/counsel_before_sun_val.json",
                        help='Path to json validation file')
    parser.add_argument('--use_cuda', type=int, default=1,
                        help='True to use cuda')
    parser.add_argument('--n_epochs', type=int, default=10,
                        help='no. of epochs to run')
    args = parser.parse_args()

    # download pretrained model
    HF_FINETUNED_MODEL = (
        "https://s3.amazonaws.com/models.huggingface.co/transfer-learning-chatbot/gpt_personachat_cache.tar.gz"  # noqa
    )

    """ Download and extract finetuned model from S3 """
    resolved_archive_file = cached_path(HF_FINETUNED_MODEL)
    tempdir = tempfile.mkdtemp()
    print("extracting archive file {} to temp dir {}".format(
        resolved_archive_file, tempdir))
    with tarfile.open(resolved_archive_file, "r:gz") as archive:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(archive, tempdir)

    # get directories
    output_dir = './saved_model'  # save models
    best_model_dir = './saved_model/best'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(best_model_dir):
        os.makedirs(best_model_dir)

    TRAIN_FILE = args.train_file
    EVAL_FILE = args.eval_file

    config_dict = {
        "batch_size": [2],
        "max_seq_length": [128],
        "learning_rate": [4e-05, 5e-6],
        "lm_coef": [3, 2],
        "max_history": [2],
        "num_candidates": [4, 2]
    }

    params = [param for param in config_dict.keys()]
    value_ls = [value_ls for value_ls in config_dict.values()]
    grid_iterlist = list(itertools.product(*value_ls))
    grid_iterlist

    grid_lm_f1_scores = {}
    grid_lm_ppl_scores = {}
    grid_lm_loss_scores = {}
    all_grid_dict = {}

    for c in range(len(grid_iterlist)):
        print(
            f"Iterating through {c+1} of {len(grid_iterlist)} grid param combinations")
        grid_param = {}
        for j, param in enumerate(params):
            grid_param[param] = grid_iterlist[c][j]
        print(f"Using parameters {grid_param}")
        all_grid_dict[c] = grid_param

        #  Put grid search parameters into arguments
        train_args = ConvAIArgs()
        train_args.overwrite_output_dir = True
        train_args.num_train_epochs = 1
        train_args.evaluate_during_training = False
        train_args.cache_dir = './cache_dir/'
        train_args.train_batch_size = grid_param['batch_size']
        train_args.eval_batch_size = grid_param['batch_size']
        train_args.max_seq_length = grid_param['max_seq_length']
        train_args.learning_rate = grid_param['learning_rate']
        train_args.lm_coef = grid_param['lm_coef']
        train_args.max_history = grid_param['max_history']
        train_args.num_candidates = grid_param['num_candidates']

        model = ConvAIModel("gpt", tempdir,
                            use_cuda=0, args=train_args)

        #  Create directory to save best model for each grid combination
        grid_best_model = './tmp/' + str(c)
        if not os.path.exists(grid_best_model):
            os.makedirs(grid_best_model)

        # Create validation loop due to bug in eval_model function. return 2 outputs instead of 3.
        N_EPOCHS = args.n_epochs
        best_loss = 999
        counter = 0
        early_stop_patience = 2
        valid_loss = []
        ppl = []
        f1 = []
        for i in range(N_EPOCHS):
            print(f'Epoch: {i}')
            print('')
            model.train_model(train_file=TRAIN_FILE)
            results = evaluation(model, eval_file=EVAL_FILE) #model.eval_model(eval_file="drive/My Drive/data/counsel_chat_250_valid.json")
            print(results)
            #https://huggingface.co/transformers/perplexity.html - ppl is equivalent to the exponentiation of the cross-entropy between the data and model predictions
            #line 868 /Users/anaconda3/envs/aiap3/lib/python3.8/site-packages/simpletransformers/language_modeling/language_modeling_model.py
            perplexity = np.exp(results['language_model_loss'])
            print(f'perplexity: {perplexity}')
            ppl.append(perplexity)
            valid_loss.append(results['language_model_loss'])
            f1.append(results['lm_f1_score'])
            if results['language_model_loss'] < best_loss:
                best_loss = results['language_model_loss']
                # save model
                print('save_model')
                model.tokenizer.save_pretrained(grid_best_model)
                model.model.save_pretrained(grid_best_model)
                model.args.save(grid_best_model)
                counter = 0
            else:
                counter += 1
            if counter == early_stop_patience:
                print('early stopping')
                break
        
        grid_lm_f1_scores[c] = f1
        grid_lm_ppl_scores[c] = ppl
        grid_lm_loss_scores[c] = best_loss

        #  Retrieve the best grid combination
        print(f"Grid lm loss {grid_lm_loss_scores}")
        best_combination_no = min(grid_lm_loss_scores.items(), key=operator.itemgetter(1))[0]
        best_combination = grid_iterlist[best_combination_no]
        best_grid_param = {}
        for j, param in enumerate(params):
            best_grid_param[param] = grid_iterlist[best_combination_no][j]
        print(f"Best parameters {best_grid_param}")

        #  Retrieve the best grid combination
        print(f"Grid lm loss {grid_lm_loss_scores}")
        best_combination_no = min(grid_lm_loss_scores.items(), key=operator.itemgetter(1))[0]
        best_combination = grid_iterlist[best_combination_no]
        best_grid_param = {}
        for j, param in enumerate(params):
            best_grid_param[param] = grid_iterlist[best_combination_no][j]
        print(f"Best parameters {best_grid_param}")

    results_df = pd.DataFrame([grid_lm_f1_scores,grid_lm_ppl_scores,grid_lm_loss_scores,all_grid_dict]).T
    results_df.columns = ['lm_f1','perplexity','loss','parameters']
    results_df.to_csv(os.path.join(output_dir,'results.csv'))

    best_model = ConvAIModel("gpt", './tmp/'+str(best_combination_no),
                          use_cuda=args.use_cuda,
                          args=train_args)

    best_model.tokenizer.save_pretrained(output_best_dir)
    best_model.model.save_pretrained(output_best_dir)
    best_model.args.save(output_best_dir)
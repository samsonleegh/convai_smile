import os
import time
import json
import logging
import random
import pandas as pd
import torch
from typing import List
from fastapi import FastAPI, Request, Form, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from simpletransformers.conv_ai import ConvAIModel, ConvAIArgs
from transformers import cached_path
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from simpletransformers.conv_ai.conv_ai_utils import get_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flask_app")

# get variables

TOPK = 0
TOPP = 0.9
TEMP = 0.6
MAX_LEN = 5

interact_args = {
    "cache_dir": "./cache_dir/",
    "max_length": MAX_LEN,
    "do_sample": True,  # sampling, False will set to greedy encoding
    "temperature": TEMP,
    "top_k": TOPK,
    "top_p": TOPP,
    "max_history": 5,
    "min_length": 1,
    "do_sample": True
}

tuned_model = ConvAIModel("gpt", "./saved_model",
                          use_cuda=False,
                          args=interact_args)

tokenizer = tuned_model.tokenizer
args = tuned_model.args
dataset = get_dataset(tokenizer, None,
                      args.cache_dir,
                      process_count=tuned_model.args.process_count,
                      proxies=tuned_model.__dict__.get("proxies", None),
                      interact=True,
                      args=args)

personalities = [dialog["personality"]
    for dataset in dataset.values() for dialog in dataset]
personality = random.choice(personalities)
global personality_decode
personality_decode = [tuned_model.tokenizer.decode(
    desc) for desc in personality]


HISTORY_FPATH = './data/history.csv'
hist_df = pd.DataFrame({'history': [],
                        'history_decode': []})
hist_df.to_csv(HISTORY_FPATH)


app = FastAPI()

app.mount(
    "/static", StaticFiles(directory="./src/static"), name="static")
templates = Jinja2Templates(directory="./src/templates")

# declare functions
class config_obj_convert:

    def __init__(self, config_dict):
        self.__dict__.update(config_dict)
        self.config_dict = config_dict

    def __str__(self):
        return str(self.config_dict)

def conv_(raw_text, history_fpath, tuned_model, args):
    """ get model response from text input and conv. history
    : raw_text param: string input
    : history param: list of strings for conv. history, use empty list if no history
    : tuned_model param: convai initialised model
    : return: out_text - string response from model, history - appended list of conv. history
    """
    hist_df = pd.read_csv(history_fpath,
                          index_col=0,
                          converters={'history': eval})  # convert string back into list
    history = list(hist_df['history'])
    history_decode = list(hist_df['history_decode'])
    history.append(tuned_model.tokenizer.encode(raw_text))
    history_decode.append(raw_text)
    used_history = history[-(2 * args.max_history + 1):]
    used_history_decode = history_decode[-(2 * args.max_history + 1):]

    print(f"personality: {personality}")
    print(f"used_history: {used_history}")
    print(f"tokenizer: {tokenizer}")
    print(f"args: {args}")

    with torch.no_grad():
        out_ids = tuned_model.sample_sequence(
            personality, used_history, tokenizer, tuned_model.model, args)
        out_text = tokenizer.decode(out_ids, skip_special_tokens=True)

    history.append(out_ids)
    history_decode.append(out_text)
    hist_df = pd.DataFrame(
        {'history': history, 'history_decode': history_decode})
    hist_df.to_csv(history_fpath)

    return out_text, used_history_decode


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request,
                                                     "enumerate": enumerate,
                                                     "topk": TOPK, 
                                                     "topp": TOPP, 
                                                     "temp": TEMP,
                                                     "max_len": MAX_LEN})


@ app.post("/conv")
def conv(request: Request, text: str=Form(...)):
    print(tuned_model.args)
    global out_text
    global used_history_decode
    out_text, used_history_decode = conv_(text, HISTORY_FPATH, tuned_model, args)

    # reverse history due to interface issue
    used_history_decode.reverse()
    return templates.TemplateResponse("index.html", {"request": request,
                                                     "reply": out_text,
                                                     "history": used_history_decode,
                                                     "personality": personality_decode,
                                                     "enumerate": enumerate,
                                                     "topk": TOPK, 
                                                     "topp": TOPP, 
                                                     "temp": TEMP,
                                                     "max_len": MAX_LEN
                                                     })

@ app.post("/setting")
def setting(request: Request, topk: int=Form(...), topp: float=Form(...), temp: float=Form(...), max_len: int=Form(...)):
    global TOPK
    TOPK = topk
    global TOPP
    TOPP = topp
    global TEMP
    TEMP = temp
    global MAX_LEN
    MAX_LEN = max_len

    print(TOPK, TOPP, TEMP, MAX_LEN)
    # reset variables
    global interact_args
    interact_args = {
        "cache_dir": "./cache_dir/",
        "max_length": MAX_LEN,
        "do_sample": True,  # sampling, False will set to greedy encoding
        "temperature": TEMP,
        "top_k": TOPK,
        "top_p": TOPP,
        "max_history": 50,
        "min_length": 1,
        "do_sample": True
    }
    global args
    args = config_obj_convert(interact_args)

    # get new personality
    personality = random.choice(personalities)
    global personality_decode
    personality_decode = [tuned_model.tokenizer.decode(
        desc) for desc in personality]

    # reset history
    HISTORY_FPATH = './data/history.csv'
    hist_df = pd.DataFrame({'history': [],
                            'history_decode': []})
    hist_df.to_csv(HISTORY_FPATH)

    return templates.TemplateResponse("index.html", {"request": request,
                                                     "reply": '',
                                                     "history": '',
                                                     "personality": personality_decode,
                                                     "enumerate": enumerate,
                                                     "topk": topk, 
                                                     "topp": topp, 
                                                     "temp": temp,
                                                     "max_len": MAX_LEN
                                                     })


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8000)

        # history = []
        # while True:
        #     raw_text = input(">>> ")
        #     while not raw_text:
        #         print("Prompt should not be empty!")
        #         raw_text = input(">>> ")
        #     history.append(tokenizer.encode(raw_text))
        #     with torch.no_grad():
        #         out_ids = self.sample_sequence(personality, history, tokenizer, model, args)
        #     history.append(out_ids)
        #     history = history[-(2 * args.max_history + 1) :]
        #     out_text = tokenizer.decode(out_ids, skip_special_tokens=True)
        #     print(out_text)
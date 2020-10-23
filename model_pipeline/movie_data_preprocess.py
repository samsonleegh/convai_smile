# references
# https://github.com/asmitakulkarni/QuoteGenerator/blob/master/01_ReadData_pdf_txt.ipynb
# https://github.com/nbertagnolli/counsel-chat/blob/master/utils.py
import logging
import argparse
import re
import json
import pdfplumber
import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Callable, List, Tuple, Optional, Union
from transformers import (AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer)
from urllib.request import urlopen
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

def pdf_to_df(pdf_file, sel_chars):
    """ Function to parse movie script pdf into dataframe
    : pdf_file param: file path for .pdf file
    : sel_chars param: characters whose dialogues will be included 
    first character will be trained as the responding character
    : return: dialogue_df dataframe made up of 3 columns consisting
    conversation 1) init, 2) response and 3) dialogue_grp
    """

    all_text = ''
    action_text = ''
    with pdfplumber.open(pdf_file) as pdf:
        # first_page = pdf.pages[4]
        # first_page_txt = first_page.extract_text()
        # print(first_page_txt)
        for pdf_page in pdf.pages:
            # extract all txt from page
            single_page_text = pdf_page.extract_text()
            # sieve starting phrases of action lines
            action_start_phrases = pdf_page.filter(lambda obj: (
                obj["object_type"] == "char" and obj["x0"] < 180)).extract_text()

            # separate each page's text with newline
            all_text = all_text + '\n' + single_page_text
            if action_start_phrases is None:
                action_start_phrases = ''
            action_text = action_text + '\n' + action_start_phrases

    # find lines to skip
    lines_to_skip = []
    for row in all_text.splitlines():
        for start_phrase in action_text.splitlines():
            # remove action lines but retain lines showing scene changes - start_phrase.upper()
            if row.startswith(start_phrase) and start_phrase != start_phrase.upper():
                lines_to_skip.append(row)

    # get non-action (dialogue) lines only
    all_lines = all_text.splitlines()
    clean_lines = [x for x in all_lines if x not in lines_to_skip]

    # read txt into pandas dataframe
    pandas_line = []
    diag_char_ls = []
    cur_line = ''
    for row in clean_lines:
        # skip page numbers
        if str.isdigit(re.sub(r"[.]","",row.strip())):
            pass
        # getting dialogue continuation by character together
        elif "(CONT’D)" in row:
            pass
        elif row.strip() == row.strip().upper() and row.strip() != '':
            diag_char_ls.append(row)
            if len(diag_char_ls) > 2:
                pandas_line.append(diag_char_ls[-2] + ':-' + cur_line)
            cur_line = ''
        elif row.strip() != row.strip().upper():
            cur_line += ' ' + row.strip()

    df = pd.DataFrame(pandas_line)
    df = df[0].str.split(':-', expand=True)
    df.columns = ['character', 'line']
    df['line'] = df.line.apply(lambda x: re.sub("[\(\[].*?[\)\]]", "", x)) # remove brackets and words in between
    
    # keep selected character dialogues and screen change prompts as dialogue groupings
    df = df[df.character.isin(sel_chars) | df.character.apply(lambda x: str.isdigit(x[0]))]
    df['dialogue_grp'] = (df['character'].apply(lambda x: str.isdigit(x[0]))).astype(int).cumsum()
    df = df[df.character.isin(sel_chars)]

    # re-initialise dialogue group after removing action scenes, or when character switches initialising dialogue
    df['dialogue_grp'] = ((df['dialogue_grp'] != df['dialogue_grp'].shift(1))| (df['character'] == df['character'].shift(1))).astype(int).cumsum()
    df = df.reset_index(drop=True)

    # remove starting line if responder initiatiates conversation first or if initiater ends conversation
    remove_responser_start = []
    for grp in df.dialogue_grp.unique():
        if df[df['dialogue_grp'] == grp].character.head(1).values[0] == sel_chars[0]:
            remove_responser_start.append(df[df['dialogue_grp'] == grp].character.head(1).index[0])
        if df[df['dialogue_grp'] == grp].character.tail(1).values[0] != sel_chars[0]:
            remove_responser_start.append(df[df['dialogue_grp'] == grp].character.tail(1).index[0])
    df = df.drop(df.index[remove_responser_start])
    # remove dialgoues with less than 2 exchanges
    df = df[df.groupby('dialogue_grp').dialogue_grp.transform('count')>=2]
    df['dialogue_grp'] = (df['dialogue_grp'] != df['dialogue_grp'].shift(1)).astype(int).cumsum()
    
    # reframe dataframe as convo response by sel_chars[0] and convo initiated by all other characters
    df_response = df[df['character'] == sel_chars[0]].reset_index(drop=True).drop(['character'],axis=1)
    df_init = df[df['character'] != sel_chars[0]].reset_index(drop=True)['line']
    dialogue_df = pd.concat([df_init,df_response], axis=1)
    dialogue_df.columns = ['init','response','dialogue_grp']

    return dialogue_df

def url_to_df(url, sel_chars):
    """ Function to convert script on url to dataframe
    : url param: url of the movie script
    : sel_chars param: characters whose dialogues will be included 
    first character will be trained as the responding character
    : return: dialogue_df dataframe made up of 3 columns consisting
    conversation 1) init, 2) response and 3) dialogue_grp
    """
    html = urlopen(url).read()
    soup = BeautifulSoup(html, features="html.parser")

    # kill all script and style elements
    for script in soup(["script", "style"]):
        script.extract()    # rip it out

    # get text
    text = soup.get_text()

    # break into lines and remove leading and trailing space on each
    lines = (line.strip() for line in text.splitlines())
    # break multi-headlines into a line each
    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
    # drop blank lines
    text = '\n'.join(chunk for chunk in chunks if chunk)

    clean_lines = []
    for row in text.splitlines():
        if row.startswith(sel_chars[0])|row.startswith(sel_chars[1])|row.startswith('Scene')|row.startswith('('):
            clean_lines.append(row)
    
    df = pd.DataFrame(clean_lines)
    df[0] = df[0].apply(lambda x: x.replace(":", ":-", 1))
    df = df[0].str.split(':-', expand=True)
    df.columns = ['character', 'line']
    # change dialogue group according to movie scene action changes
    df['dialogue_grp'] = (df['character'].apply(lambda x: x.split()[0] == 'Scene')).astype(int).cumsum()
    df = df[df.character.isin(sel_chars)]
    # remove dialogues from first scene
    df = df[df['dialogue_grp'] >= 2]
    # re-initialise dialogue group after removing action scenes, or when character switches initialising dialogue
    df['dialogue_grp'] = ((df['dialogue_grp'] != df['dialogue_grp'].shift(1))| (df['character'] == df['character'].shift(1))).astype(int).cumsum()
    df = df.reset_index(drop=True)

    # remove starting line if responder initiatiates conversation first or if initiater ends conversation
    remove_responser_start = []
    for grp in df.dialogue_grp.unique():
        if df[df['dialogue_grp'] == grp].character.head(1).values[0] == sel_chars[0]:
            remove_responser_start.append(df[df['dialogue_grp'] == grp].character.head(1).index[0])
        if df[df['dialogue_grp'] == grp].character.tail(1).values[0] != sel_chars[0]:
            remove_responser_start.append(df[df['dialogue_grp'] == grp].character.tail(1).index[0])
    remove_responser_start = remove_responser_start +[390] #,503] #text missing ':' remove manually
    df = df.drop(df.index[remove_responser_start])

    # remove dialgoues with less than 2 exchanges
    df = df[df.groupby('dialogue_grp').dialogue_grp.transform('count')>=2]
    # re-initialise dialogue groups
    df['dialogue_grp'] = (df['dialogue_grp'] != df['dialogue_grp'].shift(1)).astype(int).cumsum()
    df['line'] = df.line.apply(lambda x: re.sub("[\(\[].*?[\)\]]", "", x)) # remove brackets and words in between

    # reframe dataframe as convo response by sel_chars[0] and convo initiated by all other characters
    df_response = df[df['character'] == sel_chars[0]].reset_index(drop=True).drop(['character'],axis=1)
    df_init = df[df['character'] != sel_chars[0]].reset_index(drop=True)['line']
    dialogue_df = pd.concat([df_init,df_response], axis=1)
    dialogue_df.columns = ['init','response','dialogue_grp']
    
    return dialogue_df


def df_to_convai_data(df, personality, max_tokens, n):
    """ Function to convert dialogue dataframe into convai dataset format
    : df param: dialogue dataframe columns - init, response, dialogue_grp
    : df param: dialogue dataframe columns - init, response, dialogue_grp
    : n max_tokens: max token used from each dialogue line
    : n param: number of candidates to sample
    : return: list of dictionary dialogue in convai dataset format
    """
    
    tokenizer_class = OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained("openai-gpt")
    personality = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(traits)[:max_tokens]) for traits in personality]
    dataset = []
    
    for grp_no in df.dialogue_grp.unique():
        utterances = []
        correct_response_hist = []
        for i in range(len(df[df['dialogue_grp']==grp_no])):
            samples = sample_candidates(df, grp_no, n)
            response_raw = df[df['dialogue_grp']==grp_no]['response'].reset_index(drop=True)[i]
            # put accurate response in last of list
            samples.append(response_raw)
            candidates = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(sample)[:max_tokens]) for sample in samples]
        #     print('candidate:', candidates)
            correct_response_hist.append(candidates[-1])
            latest_init_raw = [df[df['dialogue_grp']==grp_no]['init'].reset_index(drop=True)[i]]
            latest_init = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(init)[:max_tokens]) for init in latest_init_raw]
            if i == 0:
                # get first init only
                history = latest_init
            elif i > 0:
                history.append(correct_response_hist[-2])
                history.append(latest_init[0])
        #     print('history:', history)
        #     print('')
            d = {"candidates": candidates, "history": history}
            utterances.append(copy.deepcopy(d))

        grp_dialogue = {"personality": personality,
                     "utterances": utterances}

        dataset.append(copy.deepcopy(grp_dialogue))
    
    return dataset


def sample_candidates(df, grp_no, n):
    """ Function to get sample candidate responses out of dialogue group
    : df param: dialogue dataframe columns - init, response, dialogue_grp
    : grp_no param: dialogue grp no. not to sample from
    : n param: number of candidates to sample
    : return: samples in list of string
    """
    samples = list(df[df['dialogue_grp'] != grp_no].response.sample(n))
    
    return samples

def split_data(df, test_size):
    """ Function to get train-test split according to dialogue group
    : df param: dialogue dataframe columns - init, response, dialogue_grp
    : test_size param: float proportion of test set, from 0.0 to 1.0
    : return: train, test dataframes
    """
    train_unique_idx, test_unique_idx = train_test_split(df.dialogue_grp.unique(),
                                                        test_size=test_size, 
                                                        random_state=0)
    df['split'] = np.where(df['dialogue_grp'].isin(train_unique_idx), 'train', 'val')
    df_train = df[df['split'] == 'train']
    df_val = df[df['split'] == 'val']
    
    return df_train, df_val


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf_file', type=str, default='./data/before-sunrise.pdf',
                        help='Path to movie script pdf file')
    parser.add_argument('--sel_chars_1','--list', nargs='+', default='CELINE JESSE',
                    help='selected dialogue characters from pdf')
    # parser.add_argument('--sel_chars_1', type=list, default=['CELINE','JESSE'],
    #                 help='selected dialogue characters from pdf')
    parser.add_argument('--url', type=str, default="https://sunrisesunset.fandom.com/wiki/Before_Sunset_(2004)_script",
                        help='url link to movie script')
    parser.add_argument('--sel_chars_2','--list', nargs='+', default='Céline Jesse',
                        help='selected dialogue characters from pdf')
    # parser.add_argument('--sel_chars_2', type=list, default=['Céline','Jesse'],
    #                     help='selected dialogue characters from url')
    parser.add_argument('--personality', type=list, default=['i am celine.', 'i am french.', 'i enjoy sunrise and sunset.', 'i live in paris.'],
                        help='personality of dialogue response character')
    parser.add_argument('--max_tokens', type=int, default=100,
                        help='max token used from each dialogue line')
    parser.add_argument('--n', type=int, default=3,
                        help='no. of competing candidates to train on, for dialogue response')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='float proportion of test set, from 0.0 to 1.0')

    args = parser.parse_args()

    # from model_pipeline.movie_data_preprocess_new import pdf_to_df, url_to_df, df_to_convai_data, split_data
    
    conv_df1 = pdf_to_df(args.pdf_file, args.sel_chars_1)
    conv_df2 = url_to_df(args.url, args.sel_chars_2)
    df = pd.read_csv('data/20200325_counsel_chat.csv', index_col=0)
    # train-test split
    conv_df1_train, conv_df1_test = split_data(conv_df1, test_size=args.test_size)
    conv_df2_train, conv_df2_test = split_data(conv_df2, test_size=args.test_size)

    conv_data1_train = df_to_convai_data(conv_df1_train, args.personality, args.max_tokens, args.n)
    conv_data2_train = df_to_convai_data(conv_df2_train, args.personality, args.max_tokens, args.n)

    conv_data1_test = df_to_convai_data(conv_df1_test, args.personality, args.max_tokens, args.n)
    conv_data2_test = df_to_convai_data(conv_df2_test, args.personality, args.max_tokens, args.n)
    
    # combine data
    all_train = conv_data1_train + conv_data2_train
    all_test = conv_data1_test + conv_data2_test
    print(len(all_train))
    print(len(all_test))

    with open("data/before_sun_train.json", "w") as json_file:
        json.dump(all_train, json_file)
    with open("data/before_sun_valid.json", "w") as json_file:
        json.dump(all_test, json_file)

    # consider to add therapy data

    with open('./data/counsel_chat_train.json') as json_file:
        counsel_data_train = json.load(json_file)
    with open('./data/counsel_chat_valid.json') as json_file:
        counsel_data_valid = json.load(json_file)

    counsel_all_train = all_train + counsel_data_train
    counsel_all_test = all_test + counsel_data_valid
    
    with open("data/counsel_before_sun_train.json", "w") as json_file:
        json.dump(counsel_all_train, json_file)
    with open("data/counsel_before_sun_valid.json", "w") as json_file:
        json.dump(counsel_all_test, json_file)

    

    # from model_pipeline.movie_data_preprocess_new import pdf_to_df, url_to_df, df_to_convai_data, train_test_split
    # pdf_file = './data/before-sunrise.pdf'
    # sel_chars_1 = ['CELINE','JESSE']

    # url = "https://sunrisesunset.fandom.com/wiki/Before_Sunset_(2004)_script"
    # sel_chars_2 = ['Céline','Jesse']

    # personality = ['i am celine.', 'i am french.', 'i enjoy sunrise and sunset.', 'i live in paris.']
    # max_tokens = 128

    # conv_df1 = pdf_to_df(pdf_file, sel_chars_1)
    # conv_df2 = url_to_df(url, sel_chars_2)

    # conv_data1 = df_to_convai_data(conv_df1, personality, max_tokens)
    # conv_data2 = df_to_convai_data(conv_df2, personality, max_tokens)
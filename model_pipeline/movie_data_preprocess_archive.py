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




def df_to_convai_data(dialogue_df, personality, max_tokens):
    """ Function to convert dialogue dataframe into convai data format
    : dialogue_df param: dialogue_df dataframe made up of 3 columns consisting
    conversation 1) init, 2) response and 3) dialogue_grp
    : peronality param: list of string to decribe response character personality
    : max_tokens param: integer for length of tokens used for init and response
    : return: convai_data dictionary of train and test dataset in json format
    """

    # train test split
    train_unique_idx, test_unique_idx = train_test_split(dialogue_df.dialogue_grp.unique(),
                                                        test_size=0.2, 
                                                        random_state=0)
    dialogue_df['split'] = np.where(dialogue_df['dialogue_grp'].isin(train_unique_idx), 'train', 'val')

    tokenizer_class = OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained("openai-gpt")
    tuple_map = {name: index + 1 for index, name in enumerate(dialogue_df.columns.tolist())}
    counter = 1

    train = []
    val = []
    hist_ls = []

    for row in dialogue_df.itertuples():
        d = {}
        d["utterances"] = []
        question_combined = row[tuple_map["init"]]
        true_response = row[tuple_map["response"]]

        # sample candidate response and append true response as final candidate
        candidates = sample_candidates(dialogue_df, row[tuple_map["dialogue_grp"]], "dialogue_grp", "response",
                                                3)
        candidates.append(true_response)
        if max_tokens is not None:
            # Use the provided tokenizer to tokenize the input and truncate at max_tokens
            question_combined = tokenizer.convert_tokens_to_string(
                tokenizer.tokenize(question_combined)[:max_tokens])
            candidates = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(candidate)[:max_tokens]) for
                        candidate in candidates]

        hist_ = [question_combined]+[true_response]
        d["personality"] = personality
        # append history for same dialogue
        if counter > row[tuple_map["dialogue_grp"]]:
            hist_ls = hist_ls+hist_
            d["utterances"].append({"history": hist_ls[:-1],
                            "candidates": candidates})
        # reset for new dialogue
        else: 
            d["utterances"].append({"history": [question_combined],
                            "candidates": candidates})
            hist_ls = hist_
            counter +=1
        
        if getattr(row, "split") == "train":
            train.append(d)
        elif getattr(row, "split") == "val":
            val.append(d)
    convai_data = {"train": train, "valid": val}

    return convai_data


def sample_candidates(df: pd.DataFrame, current_id: Any, id_column: str, text_column: str, n: int) -> List[str]:
    """Samples candidate responses to a question from the dataframe
    It is aware of data splits and only samples from within the same split.  This avoids
    leaking information between training validation and testing.  The sampled responses are
    also drawn from all rows which do not have the same id as the current_id
    Args:
        df: The dataframe we want to sample responses from
        current_id: The unique identifier we would like to leave out of our sampling
        id_column: The column name in the dataframe with the unique ids.  current_id should
            be an element of this column
        text_column: The column with the text we want to sample
        n: How many samples we want to take.
    Returns:
        A list of samples strings from our dataframe.
    """
    # We must only sample candidates from the correct data split to avoid information leakage across channels
    split = df[df[id_column] == current_id]["split"].tolist()[0]
    candidate_df = df[df["split"] == split]

    # Sample random rows from the dataframe not matching the current id
    sampled_texts = candidate_df[candidate_df[id_column] != current_id].sample(n)[text_column].tolist()

    # join them all
    text = " ".join(sampled_texts)

    # Replace all newlines with spaces...
    text_no_newline = re.sub("\n", " ", text).lower()

    # Split on punctuation
    split_text = re.split('[?.!]', text_no_newline)

    # Remove all empty lines
    filtered_text = [x.strip() for x in split_text if len(x.strip()) > 1]

    # Shuffle the list
    return np.random.choice(filtered_text, n).tolist()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf_file', type=str, default='./data/before-sunrise.pdf',
                        help='Path to movie script pdf file')
    parser.add_argument('--sel_chars_1', type=list, default=['CELINE','JESSE'],
                    help='selected dialogue characters from pdf')
    parser.add_argument('--url', type=str, default="https://sunrisesunset.fandom.com/wiki/Before_Sunset_(2004)_script",
                        help='url link to movie script')
    parser.add_argument('--sel_chars_2', type=list, default=['Céline','Jesse'],
                        help='selected dialogue characters from url')
    parser.add_argument('--personality', type=list, default=['i am celine', 'i am french', 'i enjoy sunrise and sunset', 'i live in paris'],
                        help='personality of dialogue response character')
    parser.add_argument('--max_tokens', type=int, default=128,
                        help='max token used from each dialogue line')

    args = parser.parse_args()

    conv_df1 = pdf_to_df(args.pdf_file, args.sel_chars_1)
    conv_df2 = url_to_df(args.url, args.sel_chars_2)
    df = pd.read_csv('data/20200325_counsel_chat.csv', index_col=0)

    conv_data1 = df_to_convai_data(conv_df1, args.personality, args.max_tokens)
    conv_data2 = df_to_convai_data(conv_df2, args.personality, args.max_tokens)

    all_data = {}
    all_data['train'] = conv_data1['train'] + conv_data2['train']
    all_data['valid'] = conv_data1['valid'] + conv_data2['valid']
    print(len(all_data['train']))

    with open("data/before_sun_train.json", "w") as json_file:
        json.dump(all_data['train'], json_file)
    with open("data/before_sun_valid.json", "w") as json_file:
        json.dump(all_data['valid'], json_file)

    # consider to add therapy data

    with open('./data/counsel_chat_250_train.json') as json_file:
        counsel_data_train = json.load(json_file)
    with open('./data/counsel_chat_250_valid.json') as json_file:
        counsel_data_valid = json.load(json_file)

    all_data['train'] = all_data['train'] + counsel_data_train
    all_data['valid'] = all_data['valid'] + counsel_data_valid
    print(len(all_data['train']))
    
    with open("data/counsel_before_sun_train.json", "w") as json_file:
        json.dump(all_data['train'], json_file)
    with open("data/counsel_before_sun_valid.json", "w") as json_file:
        json.dump(all_data['valid'], json_file)

    

    # pdf_file = './data/before-sunrise.pdf'
    # sel_chars_1 = ['CELINE','JESSE']

    # url = "https://sunrisesunset.fandom.com/wiki/Before_Sunset_(2004)_script"
    # sel_chars_2 = ['Céline','Jesse']

    # personality = ['i am celine', 'i am french', 'i enjoy sunrise and sunset', 'i live in paris']
    # max_tokens = 128

    # conv_df1 = pdf_to_df(pdf_file, sel_chars_1)
    # conv_df2 = url_to_df(url, sel_chars_2)

    # conv_data1 = df_to_convai_data(conv_df1, personality, max_tokens)
    # conv_data2 = df_to_convai_data(conv_df2, personality, max_tokens)
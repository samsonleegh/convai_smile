# reference from https://github.com/nbertagnolli/counsel-chat

"""
Preprocessing dialogue data into json format for simpletransformer training
sample format https://github.com/huggingface/transfer-learning-conv-ai/blob/master/example_entry.py
"""

import re
import json
import argparse
import numpy as np
import pandas as pd
from typing import Dict, Any, Callable, List, Tuple, Optional, Union
from transformers import (AdamW, OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer)


def convert_df_to_conv_ai_dict(df: pd.DataFrame,
                               personality: List[str],
                               response_columns: List[str],
                               tokenizer: Callable[[str], List[str]],
                               max_tokens: Optional[int] = None,
                               n_candidates: int = 6
                               ) -> Dict[str, List[Any]]:
    """
    Each entry in personachat is a dict with two keys personality and utterances, the dataset is a list of entries.
    personality:  list of strings containing the personality of the agent
    utterances: list of dictionaries, each of which has two keys which are lists of strings.
        candidates: [next_utterance_candidate_1, ..., next_utterance_candidate_19]
            The last candidate is the ground truth response observed in the conversational data
        history: [dialog_turn_0, ... dialog_turn N], where N is an odd number since the other user starts every conversation.
    Preprocessing:
        - Spaces before periods at end of sentences
        - everything lowercase
    Process each row of a DataFrame.  For each row:
    1. Grab the conversational input text
    2. Grab A the responses
    3. Create a unique data entry for each response to the question.
    4. Sample random response sentences from the dataset.
    5. Combine the random responses into a candidate list.
    Args:
        df: The counsel chat pandas dataframe
        personality: The personality we would like to use during training
        response_columns: Columns which contain valid responses to the question.  For example,
            the answerText column is the complete response of the therapist
        tokenizer: The transformers library tokenizer associated with the model we will be
            training.  It is used for setting the maximum sequence length
        max_tokens: The maximum number of tokens that any candidate, response, or question should be.
        n_candidates: The number of candidate phrases to include in the dataset for training.
            The last member of candidates is the ground truth response
    Returns:
        A dictionary with a train and validation key.
    """
    # Add one because the index of the dataframe is the 0th position.
    tuple_map = {name: index + 1 for index,
                 name in enumerate(df.columns.tolist())}

    train = []
    val = []
    # Step through every row in the dictionary
    for row in df.itertuples():

        # Get the question name and title
        # TODO:: MAKE THIS GENERAL YOU DUMB DUMB
        question_title = row[tuple_map["questionTitle"]]
        question_text = row[tuple_map["questionText"]]
        question_combined = question_title + " " + question_text

        # Step through every response column in the row
        for response_column in response_columns:

            # Get the true response
            true_response = row[tuple_map[response_column]]

            # We only want to add data if a good response exists
            if len(true_response) > 1:
                # Get candidate alternate sentances by sampling from all other questions
                candidates = sample_candidates(df, row[tuple_map["questionID"]], "questionID", "answerText",
                                               n_candidates)

                # Add the correct response to the end
                candidates.append(true_response)

                # We want to trim the size of the tokens
                if max_tokens is not None:
                    # Use the provided tokenizer to tokenize the input and truncate at max_tokens
                    question_combined = tokenizer.convert_tokens_to_string(
                        tokenizer.tokenize(question_combined)[:max_tokens])
                    candidates = [tokenizer.convert_tokens_to_string(tokenizer.tokenize(candidate)[:max_tokens]) for
                                  candidate in candidates]

                if len(candidates) != n_candidates + 1:
                    print(true_response)
                    assert False

                # Define the personality and the history
                d = {"personality": personality,
                     "utterances": [{"history": [question_combined],
                                     "candidates": candidates}]}
                if getattr(row, "split") == "train":
                    train.append(d)
                elif getattr(row, "split") == "val":
                    val.append(d)

    data = {"train": train, "valid": val}

    return data


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

    # Sample 3 random rows from the dataframe not matching the current id
    sampled_texts = candidate_df[candidate_df[id_column] != current_id].sample(
        n + 15)[text_column].tolist()

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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--max_tokens', type=int, default=200,
                        help='max token used from each dialogue line')
    parser.add_argument('--n', type=int, default=3,
                        help='no. of competing candidates to train on, for dialogue response')

    args = parser.parse_args()

    df = pd.read_csv('data/20200325_counsel_chat.csv', index_col=0)

    tokenizer_class = OpenAIGPTTokenizer
    tokenizer = tokenizer_class.from_pretrained("openai-gpt")

    d = convert_df_to_conv_ai_dict(df,
                                   [""],
                                   ["answerText"],
                                   tokenizer,
                                   max_tokens=args.max_tokens,
                                   n_candidates=args.n)

    with open("data/counsel_chat_train.json", "w") as json_file:
        json.dump(d['train'], json_file)
    # split into validation and test
    with open("data/counsel_chat_test.json", "w") as json_file:
        json.dump(d['valid'][:100], json_file)
    with open("data/counsel_chat_valid.json", "w") as json_file:
        json.dump(d['valid'][100:], json_file)
    len(d['train'])
    len(d['valid'][:100])
    len(d['valid'][100:])

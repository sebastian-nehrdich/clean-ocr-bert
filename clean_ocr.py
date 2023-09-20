# This is an OCR cleanup system that works in two stages:
# 1. Determine continuos blocks of text via next sentence prediction 
# 2. Classify whether a block of text belongs to a translation or is junk
# 3. Remove junk blocks and merge translation blocks; segment into sentences, output to file

import sys
import pandas as pd 
import numpy as np
import re
from simpletransformers.classification import ClassificationModel
from nltk.tokenize import sent_tokenize

# Load the model
model_next_sentence = ClassificationModel('bert', 'checkpoints/next-sentence', use_cuda=True)
# predict with progress bar
model_fn = ClassificationModel("roberta", "checkpoints/footnote-identification")

def clean_sentences(sentences):
    cleaned_sentences = []
    for sentence in sentences:
        add_flag = False
        if re.search(r'[a-zA-Z]', sentence):
            add_flag = True
        if len(sentence) < 5:
            add_flag = False
        if add_flag:
            cleaned_sentences.append(sentence)
    return cleaned_sentences


def detect_blocks(sentences):
    """
    Detects continuous blocks of text via next sentence prediction
    """
    sentences_for_prediction = []
    last_sentence = ""
    current_sentence = ""
    for sentence in sentences:
        sentences_for_prediction.append(last_sentence + " # " + sentence)
        last_sentence = sentence
    print("Detecting blocks of text...")
    predictions, raw_outputs = model_next_sentence.predict(sentences_for_prediction)   
    blocks = []
    current_block = []
    for sentence, prediction in zip(sentences, predictions):
        if prediction == 1:
            current_block.append(sentence)
        else:
            blocks.append(current_block)
            current_block = []
    result_sentences = []
    for block in blocks:
        merged_block = " ".join(block)        
        # split at all punctuations
        block_sentences = sent_tokenize(merged_block, language='english')
        result_sentences.extend(block_sentences)
    return result_sentences

def remove_footnotes(sentences):
    """
    Removes footnotes from a list of sentences
    """
    predictions, raw_outputs = model_fn.predict(sentences)   
    result = ""
    for sentence, prediction in zip(sentences, predictions):
        result += sentence + "\t" + str(prediction) + "\n"
        #if prediction == 0:
        #    result_sentences.append(sentence)
    return result


def clean_file(path):
    sentences = []
    with open(path, 'r') as f:
        for line in f:
            sentences.append(line.strip())
    sentences = clean_sentences(sentences)
    print("Loaded {} sentences.".format(len(sentences)))
    sentences = detect_blocks(sentences)
    result = remove_footnotes(sentences)
    with open(path + ".cleaned", 'w') as f:
        f.write(result)





def main():
    if len(sys.argv) < 2:
        print("Usage: python clean_ocr.py <path to file>")
        return
    path = sys.argv[1]
    clean_file(path)

if __name__ == "__main__":
    main()
"""Weak supervision: from ROCStories to ATOMIC knowledge tuples."""

import ast
import argparse
import csv
import itertools
import json
import os
import pickle
import random
import time
import math
import en_core_web_sm
import nltk
import numpy as np
import spacy
from tqdm import tqdm
from rouge.rouge import rouge_n_sentence_level
from nltk.stem.wordnet import WordNetLemmatizer
# local
from new_metrics_small import score_prob
from utils import read_jsonl_lines, write_items

random.seed(14)

nlp = spacy.load("en_core_web_sm")

lemm = WordNetLemmatizer()

STOP_WORDS = set(nlp.Defaults.stop_words)

parser = argparse.ArgumentParser(description="Make weak supervision data")

# Required Parameters
parser.add_argument(
        "--target_dir",
        type=str,
        help="Directory to store supervision data",
        default="distant_atomic/",
)
parser.add_argument(
        "--meta_dir",
        type=str,
        help="Directory to store supervision data",
        default="none",
)
parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Device to put model on. Options: cuda / cpu",
)

parser.add_argument(
        "--split",
        default="train",
        type=str,
        help="File split to read train or train1 / train2 if splitting the original training file",
)


parser.add_argument(
        "--kg_type",
        default="atomic",
        type=str,
        help="Atomic or ConceptNet knowledge graph",
)

parser.add_argument("--s_index", type=int,default=0)

parser.add_argument("--e_index", type=int, default=30000)

parser.add_argument("--r_path",type=str,default='weak supervision')

parser.add_argument("--debug", action="store_true", help="Debug flag")

args = parser.parse_args()


if args.kg_type == 'atomic':
   atomic_file = open('../data/v4_atomic_trn.csv') 
   atomic_reader = csv.reader(atomic_file)
   atomic_events = [row for row in atomic_reader]

   header = atomic_events[0]
   print(header)
   atomic_events = atomic_events[1:]

   # causes
   past1 = header.index("xNeed")
   past2 = header.index("xIntent")

   # effects
   future1 = header.index("xWant")
   future2 = header.index("oEffect")
   future3 = header.index("xReact")
   future4 = header.index("oWant")
   future5 = header.index("oReact")
   future6 = header.index("xEffect")

   #other 
   attr = header.index("xAttr")

   dimensions_of_interest = [
     "xNeed",
     "xIntent",
     "xWant",
     "oEffect",
     "xReact",
     "oWant",
     "oReact",
     "xEffect",
     "xAttr",
   ]

   kg = [
      [
          evt[0],
          evt[past1],
          evt[past2],
          evt[future1],
          evt[future2],
          evt[future3],
          evt[future4],
          evt[future5],
          evt[future6],
          evt[attr],
          evt[-1],
      ]
      for evt in atomic_events
   ]
   token_index = {}
   for idx, event in enumerate(kg):
       event_phrase = event[0]
       event_tokens = event_phrase.split(" ")
       for tok in event_tokens:
           if tok not in token_index:
              token_index[tok] = []
           token_index[tok].append(idx)

def rep_entity(event):
    event = event.replace("PersonX", "PERSON")
    event = event.replace("PersonY", "PERSON")
    return event

def correct_nps(n):
    n = n.split(' ')
    n = ' '.join([w for w in n if w not in STOP_WORDS])
    return n

import nltk 
is_noun = lambda pos: pos[:2] == 'NN'

def pass_filter(r,sent):
    r = nltk.word_tokenize(r)
    nouns = [w for (w,p) in nltk.pos_tag(r) if is_noun(p) and 'person' not in w.lower() and not (w in sent or w + 's' in sent or w + 'es' in sent)]
    return len(nouns) == 0

def retrieve_events(sent, nps, vps, kg_type='atomic',story=None):
    if kg_type == 'atomic':
       all_retrieved_indices = []
       for phrase in nps + vps:
           for word in phrase.split(" "):
               if word not in STOP_WORDS and word in token_index:
                  all_retrieved_indices.extend(token_index[word])
    else:
       all_retrieved_indices = []
       keys = list(token_index.keys())
       for phrase in nps:
           for word in phrase.split(" "):
               if word not in STOP_WORDS:
                  filter_ = [token_index[k][0] for k in keys if word in k.split(" ")]
                  all_retrieved_indices.extend(filter_)
    all_retrieved_indices = list(set(all_retrieved_indices))
    retrieved = [
       kg[idx] for idx in all_retrieved_indices
    ]
    if len(retrieved) > 10:
       scores = [rouge_n_sentence_level(r[0].split(), sent.split(),1)[-1] for r in retrieved]
       top_scores = np.argsort(scores)[-10:]
       retrieved = [retrieved[s] for s in top_scores]
    return retrieved

def check_empty(list_):
    lists = [i for i in list_ if len(i) > 0]
    if len(lists) == 0:
        return True
    return False

def clean_r(r):
    r = r.replace(' "', "")
    r = r.replace('"', "")
    return rep_entity(r)

def fix_splits(f):
    for l in f:
        if '.' in l and l.index('.') != len(l):
            l = [line for line in l.split('.')]

def process_gpt2(original, input_, past_=False,kg_type='atomic'):
    original = nltk.tokenize.sent_tokenize(original)
    if len(original) < 5:
       original = [l + '.' for l in ' '.join(original).split('.')]
    saved = {}
    for sent in input_:
        if not check_empty(sent[1]):
            sent_id = sent[0]
            sent = sent[1]
            for evt in sent:
                if len(evt) > 0:
                    for node in evt:
                        if len(node) > 0:
                            if kg_type == 'atomic':
                               event_label = node[0]
                               split = node[-1]
                               relations = node[1:-1]
                               relations = [ast.literal_eval(l) for l in relations]
                               relation_types = [[dimensions_of_interest[l]] * len(relations[l]) for l in range(len(relations))]
                               relations = list(itertools.chain.from_iterable(relations))
                               relation_types = list(itertools.chain.from_iterable(relation_types))
                               if sent_id in saved.keys():
                                  saved[sent_id]['relations'].extend(relations)
                                  saved[sent_id]['relation_types'].extend(relation_types)
                               else:
                                  saved[sent_id] = {'relations': relations, 'relation_types': relation_types}
                            else:
                               if sent_id in saved.keys():
                                  saved[sent_id]['relations'].append(' '.join([node[1],node[0],node[2]]))
                                  saved[sent_id]['relation_types'].append(node[0])
                               else:
                                  saved[sent_id] = {'relations': [' '.join([node[1],node[0],node[2]])], 'relation_types':[node[0]]}
        else:
            saved[sent[0]] = {'relations': [], 'relation_types':[]}
    saved_rels = {}
    s_ids = []
    all_substories, all_relations, all_relation_types, sub_lens, sent_ids = [],[], [], [], []
    for sent_id, input_ in saved.items():
        raw_relations = input_['relations']
        relation_types = input_['relation_types']
        if len(raw_relations) > 0:
           relations = [(clean_r(raw_relations[r]),relation_types[r]) for r in range(len(raw_relations))]
           relations = list(set(relations))
           relation_types = [r[1] for r in relations]
           relations = [r[0] for r in relations]
           if past_:
               substory = " ".join(original[:sent_id] + ['<|' + str(sent_id) + '|>'])
           else:
               substory = " ".join(original[:sent_id] + ['<|' + str(sent_id) + '|>'] + original[sent_id + 1 :])
           all_relations.extend(relations)
           all_substories.extend([substory] * len(relations)) 
           sub_lens.append(len(relations))
           all_relation_types.extend(relation_types)
           sent_ids.append(sent_id) 
           s_ids.extend([sent_id] * len(relations)) 

        else:
           all_substories.extend(([]))
           sub_lens.append(0)
           sent_ids.append(sent_id)
    start = time.perf_counter()
    scores = score_prob(all_substories, all_relations, all_relation_types, eval_sents=[original[sent_id] for sent_id in s_ids], kg_type=args.kg_type)
    start_idx = 0
    for idx, sent_id in enumerate(sent_ids):
        quantity = sub_lens[idx]
        end_idx = start_idx + quantity 
        sent_scores = scores[start_idx:end_idx]
        sent_relations = all_relations[start_idx:end_idx]
        sent_types = all_relation_types[start_idx:end_idx]
        for dim in dimensions_of_interest:
            sent_scores_dim = [sent_scores[s] for s in range(len(sent_scores)) if sent_types[s] == dim]
            sent_relations_dim = [sent_relations[s] for s in range(len(sent_scores)) if sent_types[s] == dim]
            top_i = np.argsort(sent_scores_dim)[-5:]
            chosen_relations = [sent_relations_dim[i] for i in top_i]
            chosen_scores = [sent_scores_dim[i] for i in top_i]
            if sent_id not in saved_rels.keys():
               saved_rels[sent_id] = {}
            if dim not in saved_rels[sent_id].keys():
               saved_rels[sent_id][dim] = {'relations':[],'scores':[]}
            chosen_relations = [chosen_relations[i] for i in range(len(chosen_relations)) if chosen_scores[i] != -math.inf]
            chosen_scores = [s for s in chosen_scores if s != -math.inf]
            saved_rels[sent_id][dim]['relations'].extend(chosen_relations)
            saved_rels[sent_id][dim]['scores'].extend(chosen_scores)
        start_idx = end_idx
    return saved_rels


def clean_sentence_names(sentence_tokens, names):
    cleaned_sentence_tokens = []

    for token in sentence_tokens:
        if token in names:
            cleaned_sentence_tokens.append("PERSON")
        else:
            cleaned_sentence_tokens.append(token)
    return cleaned_sentence_tokens

STORY_FIELDS = [
    "sentence1",
    "sentence2",
    "sentence3",
    "sentence4",
    "sentence5"
]

def format_baseline(retrievals,kg_type='atomic'):
    saved_rels = {}
    if kg_type == 'atomic':
       for i in range(len(retrievals)):
           relations = [ast.literal_eval(r) for r in retrievals[i][1][0][0][1:-1]] 
           saved_rels[i] = {}
           for d in range(len(dimensions_of_interest)):
               saved_rels[i][dimensions_of_interest[d]] = {'relations': relations[d], 'scores':[0 * len(relations[d])]}
    return retrievals


def main(args):
    stories = read_jsonl_lines('../data/' + 'h_atomic_' + args.split + '.jsonl')
    stories = stories[args.s_index:args.e_index]
    names = {l.replace("\n", "").lower() for l in open("names_list.txt").readlines()}
    for story in tqdm(stories):
        if story["storyid"] + '.jsonl' in os.listdir(args.target_dir):
           continue
        if args.debug:
            print(story)
        all_retrievals = []
        for field in STORY_FIELDS:
            clean_tokens = clean_sentence_names(story[field + "_tokens"], names)
            story[field + "_tokens"] = clean_tokens
            story[field + "_tokenized_str"] = ' '.join(clean_tokens)

        for idx, field in enumerate(STORY_FIELDS):
            vp_ = story[field + "_verb_phrases"]
            np_ = story[field + "_noun_phrases"]
            np_ = [
                  t
                  for t in np_
                  if t != "person"
                  and t != "it"
                  and t != "he"
                  and t != "she"
                  and t != "her"
                  and t != "his"
                  and t != "they"
                  and t != "their"
                  and t != "who"
                  and t != "him"
                  and t != "i"
                  and t != "what"
                  and t != "me"
                  and t not in names
                  ]

            if args.debug:
               print(vp_)
               print(np_)

            if args.kg_type == 'atomic':
               all_retrievals.append((idx, [retrieve_events(story['sentence' + str(idx +  1)], np_, vp_)]))
        if len([s for s in all_retrievals if len(s[1][0]) > 0])  < 2:
           continue
        story_sents = [story[f] + '.' * (story[f][-1] != '.') for f in STORY_FIELDS]
        if len(story_sents) < 5:
           story_sents += ['no sentence.']
        if args.kg_type == 'atomic':
           relations = process_gpt2(' '.join(story_sents), all_retrievals)
        if not os.path.exists(args.target_dir):
            os.mkdir(args.target_dir)
        story['distance_supervision_relations'] = relations
        file_path = os.path.join(args.target_dir, story['storyid'] + ".jsonl")
        write_items([json.dumps(story)], file_path)
        if args.meta_dir != "none":
            metadata = {}
            metadata['events'] = all_retrievals
            metadata_path = os.path.join(args.meta_dir, story['storyid'] + ".jsonl")       
            write_items([json.dumps(metadata)], metadata_path)

        if args.debug:
            print(story)
            print(relations)
    if args.r_path != 'weak supervision':
       write_items([json.dumps(r) for r in stories], output_file=os.path.join(args.r_path, args.split + '-ds.jsonl'))   
    else:
       write_items([json.dumps(r) for r in stories], output_file=os.path.join(args.target_dir, "baseline" * args.baseline + args.split + '-ds.jsonl'))
       if args.meta_dir != "none":
          write_items([json.dumps(r) for r in stories], output_file=os.path.join(args.meta_dir, args.split + '-ds.jsonl'))

if __name__ == "__main__":
    args = parser.parse_args()
    print("====Input Arguments====")
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")
    main(args)


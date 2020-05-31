import os
import re
import json
import pickle 
import random
import itertools

random.seed(0)

dims = {'atomic':["xNeed","xIntent","xWant","oEffect","xReact","oWant","oReact","xEffect","xAttr"]}

def process_mem(dim_id, mem):
    mem = [(r,mem[r]['relations']) for r in mem.keys() if r != dim_id]
    mem = [' <|' + m[0] + '|> ' + m[1][-1] for m in mem if len(m[1]) > 0]
    return ''.join(mem) 

def clean_mem(mem):
    mem = [(d,mem[d]['relations'],mem[d]['scores']) for d in mem.keys() if len(mem[d]['relations']) > 0]
    mem = [' <|' + m[0] + '|> ' + ' <|r|> ' + ' <|r|> '.join(m[1]) + ' <|r|> ' +  ' <|s|> ' + ' <|s|> '.join([str(s) for s in m[2]]) for m in mem]
    return mem

def process_ds(l, kg_type='atomic'):
    data = l['distance_supervision_relations']
    relations = []
    while len(relations) == 0:
          sent_id = random.sample(list(data.keys()),1)[0]
          dim_id = random.sample(dims[kg_type],1)[0]
          relations = data[sent_id][dim_id]['relations']
    relation = relations[-1]
    #mem = ''.join([' <|sent' + s + '|>' + process_mem(dim_id, data[s]) for s in data.keys() if int(s) < int(sent_id)])
    mem_vals = [clean_mem(data[s]) for s in data.keys() if int(s) < int(sent_id)]
    mem_keys = [s for s in data.keys() if int(s) < int(sent_id)]
    mem = dict(zip(mem_keys, mem_vals))
    tgts = ' <|sent' + sent_id + '|> ' + '<|' + dim_id + '|> ' + 'end_switch' + relation
    return tgts, mem

def filter_story(input_, dim_, sent_id):
    if dim_ == '<|xNeed|>' or dim_ == '<|xAttr|>' or dim_ == '<|xIntent|>':
       return input_[:sent_id+1]
    return input_

def process_ds_all(l, kg_type='atomic'):
    relations = []
    mem_vals = []
    mem_keys = []
    tgts = []
    for sent_id in range(5):
        for dim_id in range(len(dims[kg_type])):
            rel = l[str(sent_id)][dims[kg_type][dim_id]]['relations']
            if len(rel) > 0:
               rel = rel[-1]
            else:
               rel = 'none'
            relations.append(rel)
            mem_vals.append([clean_mem(l[s]) for s in l.keys() if int(s) < int(sent_id)])
            mem_keys.append([s for s in l.keys() if int(s) < int(sent_id)])
            tgts.append(' <|sent' + str(sent_id) + '|> ' + '<|' + dims[kg_type][dim_id] + '|> ' + 'end_switch' + rel)
    mem = [dict(zip(mem_keys[i], mem_vals[i])) for i in range(len(mem_keys))]
    return tgts, mem

def process_ds_half(l, kg_type='atomic'):
    relations = []
    mem_vals = []
    mem_keys = []
    tgts = []
    for sent_id in range(5):
        dim_ids = random.sample(list(range(len(dims[kg_type]))),5)
        for dim_id in dim_ids:
            rel = l[str(sent_id)][dims[kg_type][dim_id]]['relations']
            if len(rel) > 0:
               rel = rel[-1]
            else:
               rel = 'none'
            relations.append(rel)
            mem_vals.append([clean_mem(l[s]) for s in l.keys() if int(s) < int(sent_id)])
            mem_keys.append([s for s in l.keys() if int(s) < int(sent_id)])
            tgts.append(' <|sent' + str(sent_id) + '|> ' + '<|' + dims[kg_type][dim_id] + '|> ' + 'end_switch' + rel)
    mem = [dict(zip(mem_keys[i], mem_vals[i])) for i in range(len(mem_keys))]
    return tgts, mem

def _roc_stories_half(data_dir, split, comet=False, kg_type='atomic', use_filter=True):
    file_name = os.path.join(data_dir, 'c' * comet + 'h' * (1-comet) + '_' + kg_type + '_' + split + '.jsonl') 
    file = open(file_name).readlines()  
    data = [json.loads(d.strip()) for d in file]
    ids = [[d['storyid'] for i in range(25)] for d in data]
    ids = list(itertools.chain.from_iterable(ids))
    srcs = [[[d['sentence1'],d['sentence2'],d['sentence3'],d['sentence4'],d['sentence5']] for i in range(25)] for d in data]
    srcs = list(itertools.chain.from_iterable(srcs))
    outputs = [process_ds_half(d['distance_supervision_relations'],kg_type) for d in data]
    tags = [[o.split('end_switch')[0].split(' ') for o in outputs[i][0]] for i in range(len(outputs))]
    tags = list(itertools.chain.from_iterable(tags))
    tags = [[t for t in list_ if len(t) > 0] for list_ in tags]
    sent_indx = [int(t[0].replace('<|sent','').replace('|>','')) for t in tags]
    dims = [t[1] for t in tags]
    if use_filter:
       srcs = [filter_story(srcs[i],dims[i],sent_indx[i]) for i in range(len(srcs))]
    srcs = [' '.join(s) for s in srcs]
    srcs = [srcs[i] + ' ' + ' '.join(tags[i]) for i in range(len(srcs))]
    tgts = [[o.split('end_switch')[1] for o in outputs[i][0]] for i in range(len(outputs))]
    tgts = list(itertools.chain.from_iterable(tgts))
    mems = [[o for o in outputs[i][1]] for i in range(len(outputs))]
    mems = list(itertools.chain.from_iterable(mems))
    return srcs, tgts, mems, ids

def roc_stories_half(data_dir, comet=False, kg_type='atomic',use_filter=True):
    val_stories, val_infs, val_mems, val_ids = _roc_stories_half(data_dir, 'val', comet, kg_type,use_filter=use_filter)
    train_stories, train_infs, train_mems, train_ids = _roc_stories_half(data_dir, 'train',comet, kg_type,use_filter=use_filter)
    test_stories, test_infs, test_mems, test_ids = _roc_stories_half(data_dir, 'test', comet, kg_type,use_filter=use_filter)
    return (train_stories, train_infs, train_mems, train_ids), (val_stories, val_infs, val_mems, val_ids), (test_stories, test_infs, test_mems, test_ids)

def _roc_stories_all(data_dir, split, comet=False, kg_type='atomic', use_filter=True):
    file_name = os.path.join(data_dir, 'c' * comet + 'h' * (1-comet) + '_' + kg_type + '_' + split + '.jsonl') 
    file = open(file_name).readlines()  
    data = [json.loads(d.strip()) for d in file]
    ids = [[d['storyid'] for i in range(45)] for d in data]
    ids = list(itertools.chain.from_iterable(ids))
    srcs = [[[d['sentence1'],d['sentence2'],d['sentence3'],d['sentence4'],d['sentence5']] for i in range(45)] for d in data]
    srcs = list(itertools.chain.from_iterable(srcs))
    outputs = [process_ds_all(d['distance_supervision_relations'],kg_type) for d in data]
    tags = [[o.split('end_switch')[0].split(' ') for o in outputs[i][0]] for i in range(len(outputs))]
    tags = list(itertools.chain.from_iterable(tags))
    tags = [[t for t in list_ if len(t) > 0] for list_ in tags]
    sent_indx = [int(t[0].replace('<|sent','').replace('|>','')) for t in tags]
    dims = [t[1] for t in tags]
    if use_filter:
       srcs = [filter_story(srcs[i],dims[i],sent_indx[i]) for i in range(len(srcs))]
    srcs = [' '.join(s) for s in srcs]
    srcs = [srcs[i] + ' ' + ' '.join(tags[i]) for i in range(len(srcs))]
    tgts = [[o.split('end_switch')[1] for o in outputs[i][0]] for i in range(len(outputs))]
    tgts = list(itertools.chain.from_iterable(tgts))
    mems = [[o for o in outputs[i][1]] for i in range(len(outputs))]
    mems = list(itertools.chain.from_iterable(mems))
    return srcs, tgts, mems, ids

def _roc_stories(data_dir, split, comet=False, kg_type='atomic', use_filter=True):
    file_name = os.path.join(data_dir, 'c' * comet + 'h' * (1-comet) + '_' + kg_type + '_' + split + '.jsonl') 
    file = open(file_name).readlines()	
    data = [json.loads(d.strip()) for d in file]
    ids = [d['storyid'] for d in data]
    srcs = [[d['sentence1'],d['sentence2'],d['sentence3'],d['sentence4'],d['sentence5']] for d in data]
    outputs = [process_ds(d,kg_type) for d in data]
    tags = [o[0].split('end_switch')[0].split(' ') for o in outputs]
    tags = [[t for t in list_ if len(t) > 0] for list_ in tags]
    sent_indx = [int(t[0].replace('<|sent','').replace('|>','')) for t in tags]
    dims = [t[1] for t in tags]
    if use_filter:
       srcs = [filter_story(srcs[i],dims[i],sent_indx[i]) for i in range(len(srcs))]
    srcs = [' '.join(s) for s in srcs]
    srcs = [srcs[i] + outputs[i][0].split('end_switch')[0] for i in range(len(srcs))]
    tgts = [o[0].split('end_switch')[1] for o in outputs]
    mems = [o[1] for o in outputs]
    return srcs, tgts, mems, ids

def roc_stories_all(data_dir, comet=False, kg_type='atomic',use_filter=True):
    val_stories, val_infs, val_mems, val_ids = _roc_stories_all(data_dir, 'val', comet, kg_type,use_filter=use_filter)
    train_stories, train_infs, train_mems, train_ids = _roc_stories_all(data_dir, 'train',comet, kg_type,use_filter=use_filter)
    test_stories, test_infs, test_mems, test_ids = _roc_stories_all(data_dir, 'test', comet, kg_type,use_filter=use_filter)
    return (train_stories, train_infs, train_mems, train_ids), (val_stories, val_infs, val_mems, val_ids), (test_stories, test_infs, test_mems, test_ids)

def roc_stories(data_dir, comet=False, kg_type='atomic',use_filter=True):
    train_stories, train_infs, train_mems, train_ids = _roc_stories(data_dir, 'train',comet, kg_type,use_filter=use_filter)
    val_stories, val_infs, val_mems, val_ids = _roc_stories(data_dir, 'val', comet, kg_type,use_filter=use_filter)
    test_stories, test_infs, test_mems, test_ids = _roc_stories(data_dir, 'test', comet, kg_type,use_filter=use_filter)
    return (train_stories, train_infs, train_mems, train_ids), (val_stories, val_infs, val_mems, val_ids), (test_stories, test_infs, test_mems, test_ids)

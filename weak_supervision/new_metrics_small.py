import torch
import math
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_fct = torch.nn.CrossEntropyLoss(reduction='none',ignore_index=0)
model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
model.eval()
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

def add_template(rel, dim, kg_type='atomic'):
    if len(rel) == 0:
       rel = 'none.'
    if rel[-1] != '.':
       rel += '.'

    if 'xEffect' in dim: 
       return 'PersonX is likely: ' + rel 

    if 'oEffect' in dim: 
       return 'PersonY is likely: ' + rel 

    if 'xWant' in dim: 
       return 'PersonX wants: ' + rel 
    
    if 'oWant' in dim: 
       return 'PersonY wants: ' + rel

    if 'xIntent' in dim: 
       return 'PersonX wanted: ' + rel 

    if 'oIntent' in dim:
       return 'PersonY wanted: ' + rel

    if 'xAttr' in dim: 
       return 'PersonX is seen as: ' + rel

    if 'xNeed' in dim: 
       return 'PersonX needed: ' + rel 

    if 'xReact' in dim:
       return 'PersonX then feels: ' + rel 

    if 'oReact' in dim:
       return 'Others then feel: ' + rel 

    return rel 

def adjust_cands1(c):
    if len(c) > 0:
       c = c[:-1]
    return c

def adjust_tensor2(t):
    if len(t) > 0:
       t[0] = 'Ġ' + t[0]
    return t

def adjust_tensor3(t):
    if len(t) > 0:
       t[0] = 'Ġ' + t[0]
    return t

def score_prob(cands, refs, types, eval_sents=None, mask_rel=True, kg_type='atomic'):
    inputs = torch.zeros((len(cands),100))
    mask = torch.ones((len(cands),100)) 
    cands1 = [c.split('<|')[0] for c in cands]
    cands1 = [adjust_cands1(cands1[i]) for i in range(len(cands))] 
    cands2 = [c.split('<|')[1].split('|>')[1] for c in cands]
    tensor_input1 = [tokenizer.tokenize(cands1[i] + ' ' + eval_sents[i]) for i in range(len(cands))]
    tensor_input2 = [tokenizer.tokenize(add_template(refs[i], types[i], kg_type)) for i in range(len(cands))]
    tensor_input2 = [adjust_tensor2(tensor_input2[i]) for i in range(len(cands))]
    tensor_input3 = [tokenizer.tokenize(cands2[i]) for i in range(len(cands))]
    tensor_input3 = [adjust_tensor3(tensor_input3[i]) for i in range(len(cands))]

    tensor_input1 = [tokenizer.convert_tokens_to_ids(tensor_input1[i]) for i in range(len(cands))]
    tensor_input2 = [tokenizer.convert_tokens_to_ids(tensor_input2[i]) for i in range(len(cands))]
    tensor_input3 = [tokenizer.convert_tokens_to_ids(tensor_input3[i]) for i in range(len(cands))]
    tensor_input = [(tensor_input1[i] + tensor_input2[i] + tensor_input3[i])[:99] + [tokenizer.convert_tokens_to_ids(tokenizer.special_tokens_map['eos_token'])] for i in range(len(cands))]

    lengths = []
    for i in range(inputs.size(0)):
        mask[i,len(tensor_input[i]):] = 0
        if mask_rel:
           mask[i,len(tensor_input1[i]):len(tensor_input1[i])+len(tensor_input2[i])] = 0
        lengths.append(mask[i].nonzero().size(0))
        inputs[i,:len(tensor_input[i])] = torch.Tensor(tensor_input[i])
    losses = []
    steps = 0
    batch_size = 130 
    inputs = inputs.long().cuda()
    input_mask = mask.cuda()
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    while steps < inputs.size(0):
          lm_logits = model(inputs[steps:steps+batch_size,:])[0]
          shift_input_mask = input_mask[steps:steps+batch_size, 1:].contiguous().view(-1)
          shift_labels = inputs[steps:steps+batch_size, 1:].contiguous().view(-1)
          shift_logits = lm_logits[..., :-1, :].contiguous()
          shift_logits = shift_logits.view(-1, shift_logits.size(-1)) 
          loss = loss_fct(shift_logits, shift_labels)
          loss_mask = torch.mul(shift_input_mask, (shift_labels > 0).long())
          loss = torch.mul(loss_mask, loss)          
          loss = loss.view(inputs[steps:steps+batch_size,:].size(0),-1).sum(dim=1)
          loss = loss.cpu().tolist()
          per_token_loss = [-(loss[i]/lengths[steps+i]) for i in range(len(loss))]
          losses.extend(per_token_loss) 
          steps += inputs[steps:steps+batch_size,:].size(0)
    for i in range(len(refs)):
        if refs[i].lower() == 'none':
           losses[i] = -math.inf
    return losses

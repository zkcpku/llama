import json
import fire
import tqdm
import numpy as np

def read_jsonl(filename):
    with open(filename, 'r') as f:
        for line in f:
            yield json.loads(line)
def write_jsonl(filename, data):
    with open(filename, 'w') as f:
        for line in data:
            f.write(json.dumps(line) + '\n')
def write_jsonl_append(filename, each_data):
    with open(filename, 'a+') as f:
        f.write(json.dumps(each_data) + '\n')

def flatten_list(list_of_list):
    return [item for sublist in list_of_list for item in sublist]

def cal_sentence_ppl(loss_list):
    return np.exp(np.mean(loss_list))

def cal_corpus_ppl(loss_list, nonzero_token_num_list):
    all_token_loss = np.array(loss_list) * np.array(nonzero_token_num_list)
    all_token_num = np.sum(nonzero_token_num_list)
    return np.exp(np.sum(all_token_loss) / all_token_num)

def cal_sentence_acc(NextToken_accnum_list, nonzero_token_num_list):
    each_sentence_acc = np.array(NextToken_accnum_list) / np.array(nonzero_token_num_list)
    return np.mean(each_sentence_acc)

def cal_corpus_acc(NextToken_accnum_list, nonzero_token_num_list):
    return np.sum(NextToken_accnum_list) / np.sum(nonzero_token_num_list)

def cal_sentence_loss(loss_list):
    return np.mean(loss_list)
def cal_corpus_loss(loss_list, nonzero_token_num_list):
    all_token_loss = np.array(loss_list) * np.array(nonzero_token_num_list)
    all_token_num = np.sum(nonzero_token_num_list)
    return np.sum(all_token_loss) / all_token_num

def main(
    inp_path: str,
    ):
    # read jsonl
    data = read_jsonl(inp_path)
    loss_list = []
    NextToken_accnum_list = []
    nonzero_token_num_list = []
    mean_loss_list = []
    ids_list = []
    for i, each_data in enumerate(data):
        loss_list.append(each_data['loss'])
        NextToken_accnum_list.append(each_data['NextToken_accnum'])
        nonzero_token_num_list.append(each_data['nonzero_token_num'])
        mean_loss_list.append(each_data['mean_loss'])
        ids_list.append(each_data['ids'])

    loss_list = flatten_list(loss_list)
    NextToken_accnum_list = flatten_list(NextToken_accnum_list)
    nonzero_token_num_list = flatten_list(nonzero_token_num_list)
    ids_list = flatten_list(ids_list)

    sentence_ppl = cal_sentence_ppl(loss_list)
    corpus_ppl = cal_corpus_ppl(loss_list, nonzero_token_num_list)
    sentence_acc = cal_sentence_acc(NextToken_accnum_list, nonzero_token_num_list)
    corpus_acc = cal_corpus_acc(NextToken_accnum_list, nonzero_token_num_list)

    print("token_num: ", np.sum(nonzero_token_num_list))
    print("sentence_num: ", len(ids_list))
    print("====================================")
    print('sentence_ppl: ', sentence_ppl)
    print('corpus_ppl: ', corpus_ppl)
    print('sentence_acc: ', sentence_acc)
    print('corpus_acc: ', corpus_acc)
    print('sentence_loss: ', cal_sentence_loss(loss_list))
    # print(np.exp(cal_sentence_loss(loss_list)))
    print('corpus_loss: ', cal_corpus_loss(loss_list, nonzero_token_num_list))
    # print(np.exp(cal_corpus_loss(loss_list, nonzero_token_num_list)))


if __name__ == '__main__':
    fire.Fire(main)

    


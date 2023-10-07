import numpy as np
import copy
import torch
import pickle
import py_vncorenlp
import underthesea
import os 
from itertools import product
import nltk
nltk.download('punkt')

# pycorenlp_path = f""
# py_vncorenlp.download_model(save_dir=r'D:\Research\comOP\competition\VLSP2023\vncorenlp')
# word_segmentator = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=r'D:\Research\comOP\competition\VLSP2023\vncorenlp')

def parameters_to_model_name(param_dict):
    """
    :param param_dict: {"config": {}, "model": {}}
    :return:
    """
    assert "config" in param_dict, "must need config parameters."
    
    root_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    result_file, model_file =  os.path.join(root_dir, "ModelResult"), os.path.join(root_dir, "PreTrainModel")
    os.mkdir(result_file)
    os.mkdir(model_file)
    config_param = param_dict['config']
    model_param = param_dict['model'] if "model" in param_dict else None
    optimizer_param = param_dict['optimizer'] if "optimizer" in param_dict else None

    model_name = ""

    for index, (key, value) in enumerate(config_param.items()):
        model_name += str(value) if isinstance(value, int) else value
        model_name += "_" if index != len(config_param.keys()) - 1 else ""

    model_name = model_name.replace("second", "first")
    model_name = model_name.replace("test", "run")

    if not os.path.exists(os.path.join(result_file, model_name)):
        os.mkdir(os.path.join(result_file, model_name))
    if not os.path.exists(os.path.join(model_file, model_name)):
        os.mkdir(os.path.join(model_file, model_name))

    model_name += "/"
    if model_param is not None:
        model_param_col = []
        for index, (key, value) in enumerate(model_param.items()):
            if key == "first_stage" or key == "factor":
                continue

            if isinstance(value, float) or isinstance(value, int):
                value = str(int(value * 10))

            model_param_col.append(key[:4] + "_" + value)

        model_name += "_".join(model_param_col)

    result_file, model_file = os.path.join(result_file, model_name), os.path.join(model_file, model_name)

    if not os.path.exists(result_file):
        os.mkdir(result_file)
    if not os.path.exists(model_file):
        os.mkdir(model_file)

    return model_name

def read_standard_file(path):
    """
    :param: path
    :return: sentence collection, label collection
    """
    sent_col, label_col, comparative_labels = [], [], []
    last_sentence = ""
    with open(path, "r", encoding="utf-8") as f:
        for line in f.readlines():
            line = line.rstrip('\n')
            # print(type(line))
            if line[:2] == "[{":
                labels = line
                isComparative = 1
            else:
                if last_sentence != "":
                    sent_col.append(last_sentence)
                    label_col.append(labels)
                    comparative_labels.append(isComparative)
                isComparative = 0
                last_sentence = line
                labels = ""
        
        sent_col.append(last_sentence)
        label_col.append(labels)
        comparative_labels.append(isComparative)
        return sent_col, label_col, comparative_labels

def split_element(arr, split_symbol):
    inds, eles = [], []
    for item in arr:
        ele = item.split(split_symbol)[1]
        # ele = vnese_tokenize(sent_col=[ele])[0]
        # vnese_tokenize([' '.join(eles)])[0]
        eles.append(ele)
        ind = item.split(split_symbol)[0]
        inds.append(int(ind)-1)

    if len(eles) != 0:
        return inds[0], inds[-1], vnese_tokenize(sent_col=[' '.join(eles)])[0]

    return -1, -1, None


def mapping_posistion(word_tokens, word1, word2, s_index, e_index):
    s_inds, e_inds = [], []
    for i, w in enumerate(word_tokens):
        if word1 in w:
            s_inds.append(i)
        if word2 in w:
            e_inds.append(i)
    # sr_inds = sorted(s_inds, key=lambda a: abs(s_index - a))
    # er_inds = sorted(e_inds, key=lambda a: abs(e_index - a))
    cast = list(product(s_inds, e_inds))
    cast = list(filter(lambda a: a[1] >= a[0], cast))
    cast = sorted(cast, key=lambda a: abs(a[1]-a[0]))
    return cast[0]

def write_text(data_col, path):
    with open(path, 'w', encoding='utf-8') as fp:
        for i, sent in enumerate(data_col):
            fp.write("%s\n" % sent)

def write_pickle(data_dict, path):
    with open(path, "wb") as f:
        pickle.dump(data_dict, f)

def read_pickle(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def get_max_token_length(token_col):
    token_len = -1
    for index in range(len(token_col)):
        token_len = max(token_len, len(token_col[index]))
    
    return token_len

def vnese_tokenize(sent_col = None, tokenizer = None, path=None, data_type=None, token_method='standard', save = False):
    vnese_tokens = []
    if path:
        sent_col, _ = read_standard_file(path)

    for sent in sent_col:
        if token_method == 'standard':
            # vnese_tokens.append(nltk.word_tokenize(sent))
            vnese_tokens.append(sent.split())
        else:
            # word_segs = word_segmentator.word_segment(sent)   
            # word_segs = sent 
            if token_method == 'bert' and tokenizer is not None:
                # word_seg = '[SEP]'.join(word_segs)
                word_seg = "[CLS]" + sent + "[SEP]"
                vnese_tokens.append(tokenizer.tokenize(word_seg))
            elif token_method == 'vnese':
                word_seg = sent
                vnese_tokens.append(underthesea.word_tokenize(word_seg))
        
        
    if save:
        store_path = '../data/preprocess/{}_{}_tokenized.txt'.format(data_type, token_method)
        write_text(vnese_tokens, store_path)
        # vnese_tokens = np.array(vnese_tokens)
        # np.savetxt('../data/preprocess/{}.txt'.format(data_type), vnese_tokens, fmt='%s')

    return vnese_tokens

    
def token_mapping_bert(tokenizer, bert_token_col, bert_id_col, gold_token_col):
    """
    :param bert_token_col: a list of token list by BertTokenizer (with [cls] and [sep])
    :param gold_token_col: a list char list
    :return: a map: {standard_token_index: [bert_indexes]}
    """
    assert len(bert_token_col) == len(gold_token_col), "bert data length not equal to char data length"

    mapping_col = []
    for index, bert_tokens in enumerate(bert_token_col):
        seq_map, bert_index, token_index = {}, 1, 0 ## bert token except [CLS] in index 0
        seq_bert_token, seq_bert_id, seq_gold_token = bert_tokens, bert_id_col[index], gold_token_col[index]

        # print(seq_bert_token)
        # print(seq_bert_id)
        # print(seq_gold_token)
        while bert_index < len(seq_bert_token) and token_index < len(seq_gold_token):
            seq_map[token_index] = [bert_index]

            if seq_bert_id[bert_index] == tokenizer.unk_token_id:
                bert_index = bert_index + 1
                token_index = token_index + 1
                continue

            token_length = len(seq_gold_token[token_index])
            bert_length = len(seq_bert_token[bert_index])

            if seq_bert_token[bert_index].find("##") != -1:
                bert_length = len(seq_bert_token[bert_index]) - 2

            ##Tìm tất cả các subtokens bị tách ra
            while token_length > bert_length:
                bert_index = bert_index + 1
                seq_map[token_index].append(bert_index)
                bert_length += len(seq_bert_token[bert_index])

                if seq_bert_token[bert_index].find("##") != -1:
                    bert_length -= 2
            
            assert bert_length == token_length, "appear mapping error"

            token_index += 1
            bert_index += 1
        
        seq_map[token_index] = [bert_index]
        # print(seq_map)
        mapping_col.append(seq_map)
        # print(mapping_col)
    return mapping_col

def convert_elem_dict_by_mapping(label_col, mapping_col):
    """
    :param label_col: [{"entity_1": {(s_index, e_index)}}]
    :param mapping_col: {gold_token_index: [bert_index_tokens]}
    :return:
    """
    assert len(label_col) == len(mapping_col)

    convert_label_col = []
    for index, label in enumerate(label_col):
        sequence_label, sequence_map = copy.deepcopy(label), mapping_col[index]
        # print(sequence_label)
        # print(sequence_map)
        for key in sequence_label:
            sequence_label[key] = sorted(list(sequence_label[key]), key=lambda x:x[0])

            for k in range(len(sequence_label[key])):
                sequence_label[key][k] = list(sequence_label[key][k])
        
        for key, elem_position_col in sequence_label.items():
            for elem_index, elem_position in enumerate(elem_position_col):
                s_index = elem_position[0]
                e_index = elem_position[1]
                # print(elem_position, s_index, e_index)
                # print(sequence_map[0])
                if s_index == -1 or e_index == -1:
                    sequence_label[key][elem_index] = [-1, -1]
                else:
                    ## phải xem xét trường hợp tokenize thành các từ ghép
                    sequence_label[key][elem_index] = [sequence_map[s_index][0], sequence_map[e_index][-1]]
                if key == 'result':
                    sequence_label[key][elem_index].append(elem_position[-1])
        
        for key in sequence_label:
            for k in range(len(sequence_label[key])):
                sequence_label[key][k] = tuple(sequence_label[key][k])
        
        convert_label_col.append(sequence_label)

    return convert_label_col

def map_token_index_pair_to_bert_token(tuple_pair_col, mapping_col):
    """
    :param tuple_pair_col: [[(s_index, e_index) * 5]]
    :param mapping_col: {token_index: [bert_index]}
    :return: new tuple_pair_col with bert_index
    """
    convert_tuple_pair_col = []
    for index, tuple_pair in enumerate(tuple_pair_col):
        sequence_tuple_pair, sequence_map = tuple_pair, mapping_col[index]
        new_sequence_tuple_pair = []
        for pair in sequence_tuple_pair:
            new_tuple_pair = []
            for k in range(4):
                s_index = pair[k][0]
                e_index = pair[k][1]
                if s_index == -1 or e_index == -1:
                    new_tuple_pair.append((-1, -1))
                    continue
                new_s_index, new_e_index = sequence_map[s_index][0], sequence_map[e_index][-1]
                new_tuple_pair.append((new_s_index, new_e_index))
            
            new_tuple_pair.append(pair[4])
            new_sequence_tuple_pair.append(new_tuple_pair)
        convert_tuple_pair_col.append(new_sequence_tuple_pair)
    return convert_tuple_pair_col

def convert_tokens_ids(bert_tokenizer, input_tokens, data_type = 'tokens'):
    result_data = []
    if not isinstance(input_tokens, list):
        input_tokens = input_tokens.tolist()
    for seq_tokens in input_tokens:
        if data_type == 'tokens':
            result_data.append(bert_tokenizer.convert_tokens_to_ids(seq_tokens))
        else:
            result_data.append(bert_tokenizer.convert_ids_to_tokens(seq_tokens))
    return result_data

# vnese_tokenize(None, '../data/train.txt', 'train')

def update_vocab(data, elem_dict, elem_index = 0, dim = 1):
    if dim == 0:
        if data not in elem_dict:
            elem_dict[data] = elem_index
            elem_index += 1
    else:
        for d in data:
            elem_dict, elem_index = update_vocab(d, elem_dict, elem_index, dim-1)
    
    return elem_dict, elem_index

# data (token to id) or (id to token)
def transfer_data(data, convert_dict, dim=1):
    """
    :param data: a data need be convert to ids
    :param convert_dict: a dict that token => id
    :param dim: process on which dim
    :return:
    """
    data_ids = copy.deepcopy(data)
    if dim == 0:
        for i in range(len(data_ids)):
            assert data_ids[i] in convert_dict, "data error or convert dict error!"
            data_ids[i] = convert_dict[data_ids[i]]

    else:
        for i in range(len(data_ids)):
            data_ids[i] = transfer_data(data_ids[i], convert_dict, dim=dim-1)

    return data_ids

def get_tuple_pair_num(tuple_pair_col):
    """
    :param tuple_pair_col:
    :return:
    """
    pair_num, null_tuple_pair = 0, [(-1, -1)] * 5

    for index in range(len(tuple_pair_col)):
        # traverse each pair.
        for pair_index in range(len(tuple_pair_col[index])):
            # skip null tuple pair.
            if tuple_pair_col[index][pair_index] == null_tuple_pair:
                continue

            # print(tuple_pair_col[index][pair_index])
            pair_num += 1

    return pair_num

def get_mask(input_ids, dim=1):
    """
    :param input_ids: a input ids
    :param dim: create mask in which mask
    :return: a attn mask co-respond input_ids
    """
    if dim == 0:
        return len(input_ids) * [1]

    else:
        attn_mask = []
        for index in range(len(input_ids)):
            attn_mask.append(get_mask(input_ids[index], dim=dim-1))
        return attn_mask

def get_sequence_label_item(position_symbol, polarity, elem_type, special_symbol):
    """
    :param position_symbol:
    :param polarity:
    :param elem_type:
    :param special_symbol:
    :return:
    """
    POLARITY_DICT = {
        1: "COM",
        -1: "COM-",
        2: "COM+",
        3: "SUP",
        -2: "SUP-",
        4: "SUP+", 
        -3: "DIF",
        0: "EQL"
    }
    if elem_type == "result" and special_symbol:
        return position_symbol + "-" + POLARITY_DICT[polarity]
    else:
        return position_symbol


def each_elem_convert_to_multi_sequence_label(sequence_token, each_elem, elem_type, special_symbol):
    """
    :param sequence_token:
    :param each_elem: (s_index, e_index) of each element (sub, obj, ...)
    :param elem_type: norm(3 first ele) or result 
    :param special_symbol: bool
    :return:
    """
    polarity_col = []
    sequence_label = ["O"] * len(sequence_token)

    for elem_position in each_elem:
        s_index, e_index = elem_position[0], elem_position[1]

        if elem_type == "result":
            polarity = elem_position[-1]
            polarity_col.append(polarity) # predicate element đã append polarity vào cuối trước đó
        else:
            polarity = None

        ## từ đơn
        if e_index == s_index:
            sequence_label[s_index] = get_sequence_label_item("S", polarity, elem_type, special_symbol)
            continue

        sequence_label[s_index] = get_sequence_label_item("B", polarity, elem_type, special_symbol)
        sequence_label[e_index] = get_sequence_label_item("E", polarity, elem_type, special_symbol)

        for k in range(s_index + 1, e_index):
            sequence_label[k] = get_sequence_label_item("M", polarity, elem_type, special_symbol)

    return sequence_label, polarity_col


def elem_dict_convert_to_multi_sequence_label(elem_col, token_col, label_col, special_symbol=False):
    """
    :param token_col: a list of token list.
    :param label_col: a elem dict like: {elem: [(s_index, e_index)]}
    :param special_symbol: True denote using "B-NEG" sequence label system.
    :return:
    """
    elem_pair_col, polarity_col, result_sequence_label_col = [], [], []

    for index in range(len(token_col)):
        sent_multi_col = []
        for elem_index, elem in enumerate(elem_col):
            if elem_index < 3:
                sequence_label, _ = each_elem_convert_to_multi_sequence_label(
                    token_col[index], label_col[index][elem], "norm", special_symbol
                )
                sent_multi_col.append(sequence_label)

            # result may be add special symbol label system.
            else:
                sequence_label, cur_polarity = each_elem_convert_to_multi_sequence_label(
                    token_col[index], label_col[index][elem], "result", special_symbol
                )
                polarity_col.append(cur_polarity)
                result_sequence_label_col.append(sequence_label)

        elem_pair_col.append(sent_multi_col)

    return elem_pair_col, result_sequence_label_col, polarity_col

# padding data by different data_type and max_len
def padding_data(data, max_len, dim=2, padding_num=0, data_type="norm"):
    """
    :param data: a list of matrix or a list of list data
    :param max_len: integer for norm data, a tuple (n, m) for matrix
    :param dim: denote which dim will padding
    :param padding_num: padding number default is 0
    :param data_type: "norm" or "matrix"
    :return: a data of padding
    """
    assert data_type == "norm" or data_type == "matrix", "you need send truth data type, {norm or matrix}"

    if data_type == "norm":
        assert data_type == "norm" and isinstance(max_len, int), "you need sent the integer padding length"

        if dim == 0:
            return data + [padding_num] * (max_len - len(data))

        else:
            pad_data = []
            for index in range(len(data)):
                pad_data.append(
                    padding_data(data[index], max_len, dim=dim-1, padding_num=padding_num, data_type=data_type)
                )
            return pad_data

    # padding a list of matrix by max_len
    else:
        assert data_type == "matrix" and isinstance(max_len, tuple), "you need sent the tuple padding length"
        n, m = max_len

        if dim == 0:
            pad_data = [line + [padding_num] * (m - len(line)) for line in data]
            padding_length = n - len(pad_data)

            for i in range(padding_length):
                pad_data.append([padding_num] * m)

        else:
            pad_data = []
            for index in range(len(data)):
                pad_data.append(
                    padding_data(data[index], max_len, dim - 1, padding_num, data_type)
                )

        return pad_data

def get_after_pair_representation(pair_hat, representation):
    """
    :param pair_hat:
    :param representation:
    :return:
    """
    feature_dim = len(representation[0][0])

    if len(pair_hat) == 0:
        return representation

    for index in range(len(representation)):
        assert len(pair_hat[index]) == len(representation[index]), "[ERROR] Param error or Data process error."

        for pair_index in range(len(representation[index])):
            if pair_hat[index][pair_index] == 0:
                representation[index][pair_index] = [0] * feature_dim

    return representation

def generate_train_pair_data(data_representation, data_label):
    assert len(data_representation) == len(data_label), "[ERROR] Data Length Error."

    feature_dim = len(data_representation[0][0])
    final_representation, final_label = [], []

    for index in range(len(data_representation)):
        if data_representation[index] == [[0] * feature_dim]:
            continue

        for pair_index in range(len(data_representation[index])):
            final_representation.append(data_representation[index][pair_index])
            final_label.append([data_label[index][pair_index]])

    return final_representation, final_label


def create_polarity_train_data(config, tuple_pair_col, feature_out, bert_feature_out, feature_type=1):
    """
    :param config:
    :param feature_out:
    :param tuple_pair_col:
    :param bert_feature_out:
    :param feature_type:
    :return:
    """
    representation_col, polarity_col, hidden_size = [], [], 5
    encode_hidden_size = 768 if config.model_mode == "bert" else config.hidden_size * 2

    for index in range(len(tuple_pair_col)):
        for pair_index in range(len(tuple_pair_col[index])):
            each_pair_representation = []
            for elem_index in range(4):
                s, e = tuple_pair_col[index][pair_index][elem_index]
                if s == -1:
                    # 采用5维 + 768维
                    if feature_type == 0:
                        each_pair_representation.append(torch.zeros(1, hidden_size).cpu())
                        each_pair_representation.append(torch.zeros(1, encode_hidden_size).cpu())

                    # 采用 5维
                    elif feature_type == 1:
                        each_pair_representation.append(torch.zeros(1, hidden_size).cpu())

                    # 采用 768维
                    elif feature_type == 2:
                        each_pair_representation.append(torch.zeros(1, encode_hidden_size).cpu())

                else:
                    # 采用5维 + 768维
                    if feature_type == 0:
                        each_pair_representation.append(
                            torch.mean(feature_out[index][elem_index][s: e], dim=0).cpu().view(-1, hidden_size)
                        )
                        each_pair_representation.append(
                            torch.mean(bert_feature_out[index][s: e], dim=0).cpu().view(-1, encode_hidden_size)
                        )

                    # 采用 5维
                    elif feature_type == 1:
                        each_pair_representation.append(
                            torch.mean(feature_out[index][elem_index][s: e], dim=0).cpu().view(-1, hidden_size)
                        )

                    # 采用 768维
                    elif feature_type == 2:
                        each_pair_representation.append(
                            torch.mean(bert_feature_out[index][s: e], dim=0).cpu().view(-1, encode_hidden_size)
                        )

            if torch.cuda.is_available():
                cur_representation = torch.cat(each_pair_representation, dim=-1).view(-1).cpu().numpy().tolist()
            else:
                cur_representation = torch.cat(each_pair_representation, dim=-1).view(-1).numpy().tolist()

            representation_col.append(cur_representation)

            assert tuple_pair_col[index][pair_index][-1][0] in {-1, 0, 1, 2}, "[ERROR] Tuple Pair Col Error."
            polarity_col.append([tuple_pair_col[index][pair_index][-1][0] + 1])

    return representation_col, polarity_col

def get_after_pair_representation(pair_hat, representation):
    """
    :param pair_hat:
    :param representation:
    :return:
    """
    feature_dim = len(representation[0][0])

    if len(pair_hat) == 0:
        return representation

    for index in range(len(representation)):
        assert len(pair_hat[index]) == len(representation[index]), "[ERROR] Param error or Data process error."

        for pair_index in range(len(representation[index])):
            if pair_hat[index][pair_index] == 0:
                representation[index][pair_index] = [0] * feature_dim

    return representation

def invert_dict(data_dict):
    """
    :param data_dict:
    :return:
    """
    return {v: k for k, v in data_dict.items()}

# using elem_col and position_sys to get {tag: id}
def create_tag_mapping_ids(elem_col, position_sys, other_flag=True):
    """
    :param elem_col: like: ["entity_1", "entity_2", "aspect", "scale", "predicate"]
    :param position_sys: like ["B", "M", "E", "S"], ["B", "I"], ["B", "I", "E", "S"]
    :param other_flag: true denote {"O": 0}, false denote {}
    :return:
    """
    assert "".join(position_sys) in {"BMES", "BI"}, "[ERROR] position system error!"

    tags_map_ids = {"O": 0} if other_flag else {}

    if elem_col is None or len(elem_col) == 0:
        for i, pos in enumerate(position_sys):
            tags_map_ids[pos] = i + 1 if other_flag else i

    else:
        for i, elem in enumerate(elem_col):
            for j, pos in enumerate(position_sys):
                tags_map_ids[pos + "-" + elem] = i * len(position_sys) + ((j + 1) if other_flag else j)

    return tags_map_ids

def cover_rate(g_interval, p_interval, intermittent=None, proportion=True):
    """
    :param g_interval: a tuple like [s_index, e_index)
    :param p_interval: a tuple like [s_index, e_index)
    :param proportion: True: denote return proportion, False denote return length.
    :param intermittent:
    :return: proportional of cover
    """
    l_board = max(p_interval[0], g_interval[0])
    r_board = min(p_interval[1], g_interval[1])

    gold_length = (g_interval[1] - g_interval[0])
    cover_length = max(0, (r_board - l_board))

    if not proportion:
        return cover_length

    if intermittent is None:
        return cover_length / float(gold_length)
    else:
        return cover_length / intermittent

# using split symbol get a list of string.
def split_string(line, split_symbol):
    """
    :param line: a string need be split
    :param split_symbol: a string: split symbol
    :return:
    """
    return list(filter(None, line.split(split_symbol)))

def get_label_pos_tag(cur_label):
    """
    :param cur_label:
    :return:
    """
    if cur_label.find("-") == -1:
        return cur_label, "NULL"
    else:
        return split_string(cur_label, "-")

def cover_rate(g_interval, p_interval, intermittent=None, proportion=True):
    """
    :param g_interval: a tuple like [s_index, e_index)
    :param p_interval: a tuple like [s_index, e_index)
    :param proportion: True: denote return proportion, False denote return length.
    :param intermittent:
    :return: proportional of cover
    """
    l_board = max(p_interval[0], g_interval[0])
    r_board = min(p_interval[1], g_interval[1])

    gold_length = (g_interval[1] - g_interval[0])
    cover_length = max(0, (r_board - l_board))

    if not proportion:
        return cover_length

    if intermittent is None:
        return cover_length / float(gold_length)
    else:
        return cover_length / intermittent
    
# bert type data (token to id) or (id to token)
def bert_data_transfer(bert_tokenizer, input_tokens, data_type="tokens"):
    """
    :param bert_tokenizer: a object of BERTTokenizer
    :param input_tokens: a list of token or ids
    :param data_type: "tokens" denote tokens to ids, "ids" denote ids to tokens
    :return: a list of token or ids
    """
    result_data = []

    if not isinstance(input_tokens, list):
        input_tokens = input_tokens.tolist()

    for seq_tokens in input_tokens:
        if data_type == "tokens":
            result_data.append(bert_tokenizer.convert_tokens_to_ids(seq_tokens))
        else:
            result_data.append(bert_tokenizer.convert_ids_to_tokens(seq_tokens))

    return result_data

def cartesian_product(init_elem_col, add_elem_list):
    """
    :param init_elem_col: a list of [(s_index1, e_index1), (s_index2, e_index2)], length is n
    :param add_elem_list: a list of elem: (s_index3, e_index3), length is m
    :return: a list of [(s_index1, e_index1), (s_index2, e_index2), (s_index3, e_index3)], length is n * m
    """
    result_elem_data_col = []

    if len(init_elem_col) == 0:
        for add_elem in add_elem_list:
            result_elem_data_col.append([add_elem])
        return result_elem_data_col

    for index in range(len(init_elem_col)):
        for add_elem in add_elem_list:
            result_elem_data_col.append(init_elem_col[index] + [add_elem])

    return result_elem_data_col

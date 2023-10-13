import os
import numpy as np
import underthesea
import logging

from data_utils import shared_utils
from data_utils.label_parse import LabelParser

logger = logging.getLogger(__name__)


class Dataset(object):
    def __init__(self, config):
        """
        :param config: a program configure
        :return: input_ids, attn_mask, pos_ids, dep_matrix, dep_label_matrix, label_ids
        """
        self.config = config
        self.vocab, self.pos_dict = {"pad":0, "[CLS]":1, "[SEP]":2}, {"pad":0}
        self.vocab_index, self.pos_index = 5, 5
        self.token_max_len, self.char_max_len = -1, -1
        self.train_data_dict, self.dev_data_dict, self.test_data_dict = {}, {}, {}
        self.bert_tokenizer = config.bert_tokenizer
        self.elem_col = ["subject", "object", "aspect", "result"]

    def create_data_dict(self, data_path, data_type, label_path=None):
        """
        :param data_path: sentence file path
        :param label_path: label file path
        "return: a data dict with many parameters
        """
        data_dict = {}
        sent_col, label_col, comparative_label = shared_utils.read_standard_file(data_path)

        LP = LabelParser(label_col, self.elem_col, sent_col)

        ## label_col: list of {'subject': {(s_index, e_index)* num_label_per_sent}, 'object': set(), 'aspect': set(), 'opinion': {(s_index, e_index, label)}}
        ## tuple_pair_col: list of [index tuple of each element * num_label_per_sent]
        label_col, tuple_pair_col = LP.parse_sequence_label("&&", self.config.val.polarity_dict, sent_col)
        
        ## tokenize các câu:
        # "standard": sử dụng thư viện ntlk
        # "vnese": sử dụng vncorenlp + underthesea nlp
        # "bert": tokenize câu có thêm tag <cls>, <sep>
        if not os.path.exists(self.config.path.pre_process_data[data_type]):
            word_tokens = shared_utils.vnese_tokenize(sent_col=sent_col, token_method='standard')
            data_dict['standard_token'] = word_tokens
            shared_utils.write_pickle(data_dict, self.config.path.pre_process_data[data_type])
        else:
            data_dict = shared_utils.read_pickle(self.config.path.pre_process_data[data_type])
        
        self.token_max_len = max(self.token_max_len, shared_utils.get_max_token_length(data_dict['standard_token']))

        data_dict['label_col'] = label_col
        data_dict['comparative_label'] = comparative_label

        if "bert" in self.config.model_mode :
            ## add <s> and </s> into sentence (using phoBERT)
            ## add [CLS], [SEP] if using multilingual-bert
            data_dict['bert_token'] = shared_utils.vnese_tokenize(sent_col=sent_col, tokenizer=self.bert_tokenizer, token_method='bert')
            data_dict['input_ids'] = shared_utils.convert_tokens_ids(self.bert_tokenizer, data_dict['bert_token'], 'tokens')
            
            # ánh xạ các index của standard token với index của bert token
            # (standard_index : [bert_indexes])
            mapping_col = shared_utils.token_mapping_bert(self.bert_tokenizer, data_dict['bert_token'], data_dict['input_ids'], data_dict['standard_token'])
            # thay thế các index của các extracted element bằng index khi dùng bert tokenize
            label_col = shared_utils.convert_elem_dict_by_mapping(label_col, mapping_col)
            tuple_pair_col = shared_utils.map_token_index_pair_to_bert_token(tuple_pair_col, mapping_col)
            self.char_max_len = max(self.char_max_len, shared_utils.get_max_token_length(data_dict['input_ids'])) + 2
        
        ## tư xây dựng bộ vocab (thêm các tag [CLS], [PAD])
        else:   
            self.vocab, self.vocab_index = shared_utils.update_vocab(data_dict['standard_token'], self.vocab, self.vocab_index, dim=2)
            data_dict['input_ids'] = shared_utils.transfer_data(data_dict['standard_token'], self.vocab, dim=1)
            self.char_max_len = max(self.char_max_len, shared_utils.get_max_token_length(data_dict['input_ids'])) + 2
        
        data_dict['tuple_pair_col'] = tuple_pair_col
        logger.info("Convert pair number: {}".format(shared_utils.get_tuple_pair_num(data_dict['tuple_pair_col'])) )

        token_col = data_dict['standard_token'] if self.config.model_mode == 'norm' else data_dict['bert_token']
        data_dict['attn_mask'] = shared_utils.get_mask(token_col, dim=1)
        
        special_symbol = False

        data_dict['multi_label'], data_dict['result_label'], data_dict['polarity_label'] = \
        shared_utils.elem_dict_convert_to_multi_sequence_label(
            self.elem_col, token_col, label_col, special_symbol=special_symbol
        )


        ## TODO: Tag to ids, convert tag BMESO into number representation in norm_idmap
        # multi_label là list của các 3 token sequence được đánh dấu các vị trí BMESO của 3 element đầu 
        # norm_id_map = {O: 1, B: 1, M:2,, E:3, S:4}
        data_dict['multi_label'] = shared_utils.transfer_data(
        data_dict['multi_label'],
        self.config.val.norm_id_map,
        dim=2
        )

        data_dict['result_label'] = shared_utils.transfer_data(
            data_dict['result_label'],
            self.config.val.norm_id_map,
            dim=1
        )
        
        write_dict = []
        for i in range(len(label_col)):
            write_dict.append(data_dict['standard_token'][i])
            write_dict.append("Mulilang bert token: {}".format(data_dict['bert_token'][i]))
            write_dict.append("label_col: {}".format(data_dict['label_col'][i]))
            write_dict.append("Tuple_pair_col: {}".format(data_dict['tuple_pair_col'][i]))
            write_dict.append("comparative_label: {}{}".format(data_dict['comparative_label'][i], data_dict['polarity_label'][i]))
            write_dict.append("Subject, Object, Aspect: {}".format(data_dict['multi_label'][i]))
            write_dict.append("predicate element: {}".format(data_dict['result_label'][i]))
        
        shared_utils.write_text(write_dict, "../data/data_dict/{}_dict.txt".format(data_type))

        return data_dict
        

    def generate_data(self):
        self.train_data_dict = self.create_data_dict(
            self.config.path.standard_path['train'],
            "train"
        )

        self.dev_data_dict = self.create_data_dict(
            self.config.path.standard_path['dev'],
            "dev"
        )

        self.test_data_dict = self.create_data_dict(
            self.config.path.standard_path['test'],
            "test"
        )

        self.train_data_dict = self.padding_data_dict(self.train_data_dict)
        self.dev_data_dict = self.padding_data_dict(self.dev_data_dict)
        self.test_data_dict = self.padding_data_dict(self.test_data_dict)

        self.train_data_dict = self.data_dict_to_numpy(self.train_data_dict)
        self.dev_data_dict = self.data_dict_to_numpy(self.dev_data_dict)
        self.test_data_dict = self.data_dict_to_numpy(self.test_data_dict)

    def padding_data_dict(self, data_dict):
        """
        :param data_dict:
        :return:
        """
        pad_key_ids = {0: ["input_ids", "attn_mask", "result_label"],
                       1: ["multi_label"]}

        cur_max_len = self.char_max_len

        param = [{"max_len": cur_max_len, "dim": 1, "pad_num": 0, "data_type": "norm"},
                 {"max_len": cur_max_len, "dim": 2, "pad_num": 0, "data_type": "norm"}]

        for index, key_col in pad_key_ids.items():
            for key in key_col:
                data_dict[key] = shared_utils.padding_data(
                    data_dict[key],
                    max_len=param[index]['max_len'],
                    dim=param[index]['dim'],
                    padding_num=param[index]['pad_num'],
                    data_type=param[index]['data_type']
                )

        return data_dict

    @staticmethod
    def data_dict_to_numpy(data_dict):
        """
        :param data_dict:
        :return:
        """
        key_col = ["input_ids", "attn_mask", "tuple_pair_col", "result_label", "multi_label", "comparative_label"]

        for key in key_col:
            data_dict[key] = np.array(data_dict[key])
            print(key, data_dict[key].shape)

        data_dict['comparative_label'] = np.array(data_dict['comparative_label']).reshape(-1, 1)

        return data_dict



            
            



        
    
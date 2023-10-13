from data_utils import shared_utils 
import re
from ast import literal_eval

# LABEL_CONVERT = {
#     "COM": 1,
#     "COM-":-1,
#     "COM+": 2,
#     "SUP": 3,
#     "SUP-":-2,
#     "SUP+": 4, 
#     "DIF":-3,
#     "EQL": 0
# }

class LabelParser(object):
    def __init__(self, label_col, elem_col, word_tokens, intermittent=False):
        """
        :param label_col:
        :param elem_col: ["subject", "object", "aspect", "result"]
        :param intermittent: True denote "result" using intermittent representation
        """
        self.label_col = label_col
        self.elem_col = elem_col
        self.intermittent = intermittent
        self.word_tokens = word_tokens

    def parse_sequence_label(self, split_symbol="&&", polarity_dict = None, sent_col=None):
        """
        :param split_symbol:
        :param sent_col:
        :param file_type
        :return: tuple_pair_col: format [{"ele1": {(s_index, e_index)}}]
                element representation: format {'subject': set(), 'object': set(), 'aspect': set(), 'opinion': set()}
        """
        null_label = "[{}]"
        tuple_pair_col, elem_representation = [], []
        for label_index, label in enumerate(self.label_col):
            if label == null_label:
                tuple_pair_col.append([[(-1, -1)]* 5]) ## list of one list contain 5 tuple (-1, -1)
                elem_representation.append(self.init_label_representation())
            
            else:
                global_elem_col = self.init_label_representation() ##global_elem_col: {'subject': {(s_index, e_index)* num_label_per_sent}, 'object': set(), 'aspect': set(), 'opinion': {(s_index, e_index, label)}}
                sequence_tup_pair = [] 
                label_dicts = re.findall(r'\{.*?\}', label)
                
                for pair_index, pair in enumerate(label_dicts):
                    global_elem_col, cur_tuple_pair = self.parse_each_pair_label(literal_eval(pair), global_elem_col, split_symbol, self.word_tokens[label_index], polarity_dict, sent_col[label_index])
                    sequence_tup_pair.append(cur_tuple_pair)

                tuple_pair_col.append(sequence_tup_pair)
                elem_representation.append(global_elem_col)
        return elem_representation, tuple_pair_col
                    
    def parse_each_pair_label(self, label_dict, global_elem_col, split_symbol, word_tokens, polarity_dict, sent=None):
        """
        :param sequence_label:
        :param global_elem_col:
        :param split_symbol:
        :param sent:
        :param file_type:
        :return: 
            global_elem_col: {'subject': {(s_index, e_index)* num_label_per_sent}, 'object': set(), 'aspect': set(), 'opinion': {(s_index, e_index, label)}}
            tuple_pair_representation: [(s_index, e_index)*4, (label, label)]
        """
        tuple_pair_representation, opininion_elem = [], []
        for key in label_dict:
            elem_tuple = ()
            if key != 'label':
                s_index, e_index, element = shared_utils.split_element(label_dict[key], split_symbol)
                # if element:
                #     # print(element)
                #     s_index, e_index = shared_utils.mapping_posistion(word_tokens, element[0], element[-1], s_index, e_index)
                
                elem_tuple += (s_index, e_index)

                if key == 'predicate':
                    opininion_elem += [s_index, e_index]
                    # if [s_index, e_index] == [-1, -1]:
                    #     print(label_dict)
                
                else:
                    global_elem_col[key].add(elem_tuple)
            
            else:
                label = label_dict['label']
                label = polarity_dict[label]
                elem_tuple += (label, label)

                if len(opininion_elem) == 0:
                    opininion_elem = [-1, -1]

                opininion_elem.append(label) ## [s_index_opinion, e_index_opinion, label]

            tuple_pair_representation.append(elem_tuple)
        
        global_elem_col['result'].add(tuple(opininion_elem))
        
        return global_elem_col, tuple_pair_representation

    
    def init_label_representation(self):
        return {key: set() for key in self.elem_col} ##{'subject': set(), 'object': set(), 'aspect': set(), 'opinion': set()}
    
    def init_label_representation(self):
        return {key: set() for key in self.elem_col} ##{'subject': set(), 'object': set(), 'aspect': set(), 'opinion': set()}

# sent_col, label_col = shared_utils.read_standard_file('D:/Research/comOP/competition/VLSP2023/data/train.txt')
# elem_col = ["subject", "object", "aspect", "predicate"]

# shared_utils.write_text(label_col, 'D:/Research/comOP/competition/VLSP2023/data/preprocess/train_label_col.txt')
# word_tokens = shared_utils.vnese_tokenize(sent_col=sent_col, data_type='train', save=True)
# LP = LabelParser(label_col, elem_col, word_tokens)
# label_col, tuple_pair_col = LP.parse_sequence_label("&&", sent_col)

# shared_utils.write_text(label_col, 'D:/Research/comOP/competition/VLSP2023/data/preprocess/train_label_dict.txt' )
# shared_utils.write_text(sent_col, 'D:/Research/comOP/competition/VLSP2023/data/preprocess/train_sent_col.txt' )


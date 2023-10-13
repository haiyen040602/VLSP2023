from data_utils import shared_utils
from eval_utils import eval_shared_modules
from transformers import AutoTokenizer

class BaseConfig(object):
    def __init__(self, args):
        self.epochs = args.epoch
        self.batch_size = args.batch
        self.device = args.device
        self.fold = args.fold

        self.input_size = args.input_size
        self.hidden_size = args.hidden_size
        self.num_layers = args.num_layers

        self.model_mode = args.model_mode
        self.model_type = args.model_type
        self.file_type = args.file_type
        self.stage_model = args.stage_model
        self.program_mode = args.program_mode
        self.position_sys = args.position_sys

        self.path = PathConfig(self.device, self.program_mode)
        self.val = GlobalConfig(self.position_sys)

        if "bert" in self.model_mode:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(self.model_mode)
        
        
class PathConfig(object):
    def __init__(self, file_type, program_mode):
        self.standard_path = {
            "train": "../data/train.txt",
            "test": "../data/test.txt",
            "dev": "../data/dev.txt"
        }

        self.bert_model_path = "bert-base-multilingual-cased"
        self.pre_process_path = "../data/preprocess/"
        self.pre_process_data = {
            "train": "../data/preprocess/train_data.pkl",
            "dev": "../data/preprocess/dev_data.pkl",
            "test": "../data/preprocess/test_data.pkl"
        }

class GlobalConfig(object):
    def __init__(self, position_sys):
        self.elem_col = ['subject', 'object', 'aspect', 'result']
        self.polarity_col = ['COM', 'COM+', 'COM-', 'SUP', 'SUP+', 'SUP-', 'EQL', 'DIF']
        # self.polarity_dict = {k: index for index, k in enumerate(self.polarity_col)}
        self.polarity_dict = {k: index for index, k in enumerate(self.polarity_col)}
        # self.polarity_dict ={
        #     "COM": 1,
        #     "COM-":-1,
        #     "COM+": 2,
        #     "SUP": 3,
        #     "SUP-":-2,
        #     "SUP+": 4, 
        #     "DIF":-3,
        #     "EQL": 0
        # }
        if position_sys == 'SPAN': # include (s_index, e_index)
            self.position_sys = []
        else: 
            self.position_sys = list(position_sys) # BIEOS = [B, I, E, O, S]
        
        self.special_id_map, self.norm_id_map = {"O": 0}, {"O": 0}

        # other flag is "O"
        # Tạo dictionmary chứa các pos_sys-element và giá trị số (như key-value) của nó
        # {'O': 0, 'B': 1, 'M': 2, 'E': 3, 'S': 4}
        self.norm_id_map = eval_shared_modules.create_tag_mapping_ids([], self.position_sys, other_flag=True)
         # {'O': 0, 'B-entity_1': 1, 'M-entity_1': 2, 'E-entity_1': 3, 'S-entity_1': 4, 'B-entity_2': 5, 'M-entity_2': 6, so on ...}
        self.special_id_map = eval_shared_modules.create_tag_mapping_ids(self.polarity_col, self.position_sys, other_flag=True)

        # đảo chiều dictionnary ở trên value: key
        self.invert_special_id_map = {v: k for k, v in self.special_id_map.items()}
        self.invert_norm_id_map = {v: k for k, v in self.norm_id_map.items()}


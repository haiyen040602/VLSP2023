import csv
import copy
import torch
import numpy as np
import eval_utils.eval_shared_modules as eval_shared_modules
import logging
logger = logging.getLogger(__name__)


class BaseEvaluation(object):
    def __init__(self, config, elem_col=None, ids_to_tags=None, fold=0, save_model=False):
        """
        :param config: program config table.
        :param elem_col: ["entity_1", "entity_2", "aspect", "scale", "predicate"].
        :param ids_to_tags: {0: "O", 1: "B-entity_1"}.
        :param save_model: True denote save model by optimize exact measure.
        """
        self.config = config
        self.elem_col = elem_col
        # ["subject", "object", "aspect", "result"]
        self.fold = fold

        if ids_to_tags is None: # return dict {value: pos_key} {1 : B_entity_1}
            self.ids_to_tags = eval_shared_modules.invert_dict(
                eval_shared_modules.create_tag_mapping_ids(elem_col, config.position_sys, other_flag=True)
            )
        else:
            self.ids_to_tags = ids_to_tags

        self.save_model = save_model

        self.bert_tokenizer = self.config.bert_tokenizer

        # store predict out.
        self.elem_hat = []
        self.result_hat = []
        self.predict_dict = {}

        # store optimize measure.
        self.optimize_exact_measure = {}
        self.optimize_prop_measure = {}
        self.optimize_binary_measure = {}

        # store average measure.
        self.avg_exact_measure =  {} #self.init_elem()
        self.avg_prop_measure = {}
        self.avg_binary_measure = {}

    ####################################################################################################################
    # return (s_index, e_index, label_id)
    ####################################################################################################################
    def get_elem_representation(self, s_index, e_index, cur_emotion):
        """
        :param s_index:
        :param e_index:
        :param cur_emotion:
        :return:
        """
        elem_representation = [s_index, e_index]
        if cur_emotion == "NULL":
            return tuple(elem_representation)

        ## get label id of comparative label
        emotion_ids = self.config.val.polarity_dict[cur_emotion]
        elem_representation.append(emotion_ids)

        return tuple(elem_representation) #(s_index, e_index, label_id or NUll)
    
    ##
    def sequence_label_convert_dict(self, sequence_label, pre_elem_dict, cur_tag):
        """
        :param sequence_label:
        :param pre_elem_dict:
        :param cur_tag:
        :param label_type:
        :return:
        """
        ## invert_norm_id_map = {0: O, 1: B, 2:  E, 3: S, 4: O}
        ids_mapping_dict = self.config.val.invert_norm_id_map

        if cur_tag not in pre_elem_dict:
            pre_elem_dict[cur_tag] = []

        s_index = -1
        for index in range(len(sequence_label)):
            assert sequence_label[index] in ids_mapping_dict, "ids to elem error!"

            current_tag = ids_mapping_dict[sequence_label[index]]

            # divide "B-opinion?" to B and subject
            cur_position, cur_emotion = eval_shared_modules.get_label_pos_tag(current_tag)

            # "O" denote "Others", skip this token
            if cur_position == "O":
                continue

            # "S" means alone element
            # get_elem_represenatation return (s_index, e_index) for elem, (s_index, e_index, label_ids) for result 
            # (index, index + 1, label_id?)
            if cur_position == "S":
                s_index, pre_tag = -1, ""
                pre_elem_dict[cur_tag].append(self.get_elem_representation(index, index + 1, cur_emotion))
            #(s_index=index)
            elif cur_position == "B":
                s_index = index
            # (s_index = B-index, e_index = index + 1)
            elif cur_position == "E" and s_index != -1:
                pre_elem_dict[cur_tag].append(self.get_elem_representation(s_index, index + 1, cur_emotion))
                s_index = -1


        return pre_elem_dict #[(s_index, e_index, label_ids)] of each B, E position in sequence label?

    ## trả về danh sách các vị trí B, E trong sequence của từng element
    ## format {"result": [(B_sindex, B_eindex, label_id), (E_sindex, E_e_index)], "other element": []}
    def get_elem_dict(self, target):
        """
        :param target: a list of list, each sentences have correspond label
        :return:  a list of dict like: [{"result": {s_index: length}, "elem2": {s_index: length}}]
        """
        elem_col = []

        elem_label_ids, result_label_ids = target
        for i in range(len(result_label_ids)):
            seq_elem = self.sequence_label_convert_dict(
                result_label_ids[i], {}, "result"
            )
            elem_col.append(seq_elem)

        assert len(elem_col) == len(elem_label_ids), "label length error!"

        elem_key = ["subject", "object", "aspect", "result"]

        for i in range(len(elem_col)):
            for j in range(len(elem_label_ids[i])):
                elem_col[i] = self.sequence_label_convert_dict(
                    elem_label_ids[i][j], elem_col[i], elem_key[j]
                )

        # for i in range(5):
        #     logger.info("Element dictionary: {}".format(elem_col[i]))
            
        return elem_col

    def add_data(self, elem_output, result_output, attn_mask):
        """
        :param result_output:
        :param elem_output:
        :param attn_mask:
        :return:
        """
        # add result sequence label.
        for index in range(result_output.size(0)):
            if self.config.device == "cpu":
                mask_output = result_output[index][attn_mask[index] == 1].numpy().tolist()
            else:
                mask_output = result_output[index][attn_mask[index] == 1].cpu().numpy().tolist()
            self.result_hat.append(mask_output)

        # add multi-sequence label.
        for index in range(elem_output.size(0)):
            cur_elem_hat = []
            for k in range(elem_output.size(1)):
                if self.config.device == "cpu":
                    mask_output = elem_output[index][k][attn_mask[index] == 1].numpy().tolist()
                else:
                    mask_output = elem_output[index][k][attn_mask[index] == 1].cpu().numpy().tolist()
                cur_elem_hat.append(mask_output)

            self.elem_hat.append(cur_elem_hat)

    def get_ground_truth_num(self, data_dict):
        """
        :param data_dict: {"elem": [(s_index, e_index)]}
        :return:
        """
        gold_num = {key: 0 for key in self.config.val.elem_col}

        for index in range(len(data_dict)):
            for elem in self.config.val.elem_col:
                gold_num[elem] += len(data_dict[index][elem])

        return gold_num

    ####################################################################################################################
    # each elem measure. (exact / proportional / binary)
    ####################################################################################################################
    @staticmethod
    def is_position_tuple_equal(a, b):
        """
        :param a:
        :param b:
        :return:
        """
        if a[0] == b[0] and a[1] == b[1]:
            return True
        return False

    def get_elem_num(self, gold_elem_dict, predict_elem_dict):
        """
        :param gold_elem_dict:
        :param predict_elem_dict:
        :return:
        """
        exact_num = self.get_exact_num(gold_elem_dict, predict_elem_dict)
        prop_num = self.get_cover_num(gold_elem_dict, predict_elem_dict, "prop")
        binary_num = self.get_cover_num(gold_elem_dict, predict_elem_dict, "binary")

        return exact_num, prop_num, binary_num

    def get_exact_num(self, gold_col, predict_col):
        """
        :param gold_col: [(s_index, e_index)]
        :param predict_col: [(s_index, e_index)]
        :param elem_type:
        :return: correct_num: number of correct element position prediction
        """
        correct_num = 0.0

        result_polarity_correct_num, vis, polarity_correct_num_col = 0.0, set(), [0] * 4

        if len(predict_col) == 0:
            return [correct_num, result_polarity_correct_num, polarity_correct_num_col]
        for pi in range(len(predict_col)):
            for gi in range(len(gold_col)):
                if gi in vis:
                    continue
                
                ## check if s_index, e_index in predict_dict and gold_dict are equal
                if self.is_position_tuple_equal(predict_col[pi], gold_col[gi]):
                    correct_num = correct_num + 1
                    vis.add(gi)

                    # calculate result element with polarity
                    # last element tuple has (s_index, e_index, label_id)
                    if len(predict_col[pi]) == 3 and predict_col[pi] == gold_col[gi]:
                        result_polarity_correct_num += 1
                        polarity_correct_num_col[predict_col[pi][2] + 1] += 1

                    break

        return [correct_num, result_polarity_correct_num, polarity_correct_num_col]

    @staticmethod
    ## tính tỉ lệ dự đoán đúng (độ dài câu dự đoán) / độ dài element thực tế
    def get_cover_num(gold_col, predict_col, measure_type="binary"):
        """
        :param gold_col: [(s_index, e_index)]
        :param predict_col: [(s_index, e_index)]
        :param measure_type: "prop" or "binary"
        :return: correct extract elements.
        """
        correct_num = 0.0
        result_polarity_correct_num, vis, polarity_correct_num_col = 0.0, set(), [0] * 4

        if len(predict_col) == 0:
            return [correct_num, result_polarity_correct_num, polarity_correct_num_col]

        for pi in range(len(predict_col)):
            for gi in range(len(gold_col)):
                # skip used gold index
                if gi in vis:
                    continue

                cover_prop = eval_shared_modules.cover_rate(gold_col[gi], predict_col[pi], proportion=True)

                if cover_prop > 0 and measure_type == "binary":
                    vis.add(gi)
                    correct_num += 1

                    # calculate result element with polarity
                    if len(predict_col[pi]) == 3 and predict_col[pi][2] == gold_col[gi][2]:
                        result_polarity_correct_num += 1
                        polarity_correct_num_col[predict_col[pi][2] + 1] += 1
                    break

                elif cover_prop > 0 and measure_type == "prop":
                    vis.add(gi)
                    correct_num += cover_prop

                    # calculate result element with polarity
                    if len(predict_col[pi]) == 3 and predict_col[pi][2] == gold_col[gi][2]:
                        result_polarity_correct_num += cover_prop
                        polarity_correct_num_col[predict_col[pi][2] + 1] += 1
                    break

        return [correct_num, result_polarity_correct_num, polarity_correct_num_col]

    def get_f_score(self, gold_num, predict_num, correct_num, multi_elem_score=False):
        """
        :param gold_num: {elem: number......}
        :param predict_num: {elem: number....}
        :param correct_num: {elem: number.....}
        :param multi_elem_score: True denote calculate macro and micro, False denote don't calculate.
        :return: {elem: {P:num, R:num, F:num}.....}
        """
        result_dict = {}

        # maybe elem_col and "pair"
        for elem in predict_num.keys():
            try:
                precision = correct_num[elem] / float(predict_num[elem]) * 100
            except ZeroDivisionError:
                precision = 0.0

            try:
                recall = correct_num[elem] / float(gold_num[elem]) * 100
            except ZeroDivisionError:
                recall = 0.0

            try:
                f_score = 2 * precision * recall / (precision + recall)
            except ZeroDivisionError:
                f_score = 0.0

            result_dict[elem] = {"P": precision, "R": recall, "F": f_score}

        if not multi_elem_score:
            return result_dict

        # base_elem_col = ["subject", "object", "aspect", "result"]
        base_elem_col = self.elem_col

        result_dict = self.get_macro_measure(result_dict, base_elem_col, elem_name="macro")
        result_dict = self.get_micro_measure(
            result_dict, gold_num, predict_num, correct_num, base_elem_col, elem_name="micro"
        )

        return result_dict

    def get_macro_measure(self, result_dict, multi_key_col, elem_name="macro"):
        """
        :param result_dict:
        :param multi_key_col:
        :param elem_name:
        :return:
        """
        # calculate macro-average F-Measure
        macro_precision, macro_recall, macro_f_score = 0.0, 0.0, 0.0

        for key in multi_key_col:
            macro_precision += result_dict[key]['P']
            macro_recall += result_dict[key]['R']
        macro_precision = macro_precision / float(len(self.elem_col))
        macro_recall = macro_recall / float(len(self.elem_col))

        try:
            macro_f_score = 2 * macro_precision * macro_recall / (macro_precision + macro_recall)
        except ZeroDivisionError:
            macro_f_score = 0.0

        result_dict[elem_name] = {"P": macro_precision, "R": macro_recall, "F": macro_f_score}

        return result_dict

    @staticmethod
    def get_micro_measure(result_dict, gold_num, predict_num, correct_num, multi_key_col, elem_name="micro"):
        """
        :param result_dict:
        :param gold_num:
        :param predict_num:
        :param correct_num:
        :param multi_key_col:
        :param elem_name:
        :return:
        """
        # calculate micro-average F-Measure
        micro_gold_num, micro_predict_num, micro_correct_num = 0.0, 0.0, 0.0

        for elem in multi_key_col:
            micro_gold_num += gold_num[elem]
            micro_predict_num += predict_num[elem]
            micro_correct_num += correct_num[elem]

        try:
            micro_precision = micro_correct_num / float(micro_predict_num) * 100
        except ZeroDivisionError:
            micro_precision = 0.0

        try:
            micro_recall = micro_correct_num / float(micro_gold_num) * 100
        except ZeroDivisionError:
            micro_recall = 0.0

        try:
            micro_f_score = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)
        except ZeroDivisionError:
            micro_f_score = 0.0

        result_dict[elem_name] = {"P": micro_precision, "R": micro_recall, "F": micro_f_score}

        return result_dict

    def init_elem(self, key_col=None):
        if key_col is None:
            return {elem: 0.0 for elem in self.elem_col} if self.elem_col is not None else {}
        return {elem: 0.0 for elem in key_col}

    @staticmethod
    def is_optimize(optimize_measure, measure, multi_measure_type):
        """
        :param optimize_measure: {elem: {"P": num, "R": num, "F": num}} or {}
        :param measure: {elem: {"P": num, "R": num, "F": num}}
        :param multi_measure_type: "macro" or "micro"
        :return:
        """
        if len(optimize_measure) == 0 or optimize_measure[multi_measure_type]['F'] < measure[multi_measure_type]['F']:
            return True
        return False

    ####################################################################################################################
    # calculate pair measure. (exact / proportional / binary)
    ####################################################################################################################
    def get_pair_num(self, gold_pair_col, predict_pair_col, polarity=False):
        """
        :param gold_pair_col: [n, tuple_pair]
        :param predict_pair_col: [n, tuple_pair]
        :param polarity:
        :return:
        """
        # logger.info("Gold pair: {} - Predict pair: {}".format(gold_pair_col, predict_pair_col))

        exact_num = self.get_exact_pair_num(gold_pair_col, predict_pair_col, polarity)
        prop_num = self.get_cover_pair_num(gold_pair_col, predict_pair_col, "prop", polarity)
        binary_num = self.get_cover_pair_num(gold_pair_col, predict_pair_col, "binary", polarity)
        return exact_num, prop_num, binary_num

    @staticmethod
    def get_exact_pair_num(gold_col, predict_col, polarity=False):
        """
        :param gold_col: [gold_pair_num, tuple_pair]
        :param predict_col: [predict_pair_num, tuple_pair]
        :param polarity
        :return: correct_num.
        """
        correct_num, null_pair = 0.0, [(-1, -1)] * 5

        for gold_index in range(len(gold_col)):
            if gold_col[gold_index] == null_pair:
                continue

            for predict_index in range(len(predict_col)):
                if polarity and gold_col[gold_index] == predict_col[predict_index]:
                    correct_num += 1
                    break
                elif not polarity and gold_col[gold_index][: -1] == predict_col[predict_index]:
                    correct_num += 1
                    break

        return correct_num

    def get_cover_pair_num(self, gold_col, predict_col, measure_type="binary", polarity=False):
        """
        :param gold_col:
        :param predict_col:
        :param measure_type:
        :param polarity:
        :return:
        """
        correct_num, null_pair = 0.0, [(-1, -1)] * 5

        for gold_index in range(len(gold_col)):
            if gold_col[gold_index] == null_pair:
                continue

            for predict_index in range(len(predict_col)):
                is_pair, cover_prop = self.pair_is_cover(gold_col[gold_index], predict_col[predict_index], polarity)

                if is_pair:
                    correct_num = correct_num + 1 if measure_type == "binary" else correct_num + cover_prop
                    break

        return correct_num

    @staticmethod
    def pair_is_cover(gold_tuple_pair, predict_tuple_pair, polarity=False):
        """
        :param gold_tuple_pair: [(s_index, e_index)]
        :param predict_tuple_pair: [(s_index, e_index)]
        :param polarity: False denote without polarity, True denote with polarity
        :return:
        """
        null_elem = (-1, -1)
        gold_elem_length, cover_elem_length = 0, 0

        for index in range(4):
            if gold_tuple_pair[index] == null_elem and predict_tuple_pair[index] == null_elem:
                continue

            cur_gold_length = gold_tuple_pair[index][1] - gold_tuple_pair[index][0]
            cur_cover_length = eval_shared_modules.cover_rate(
                gold_tuple_pair[index], predict_tuple_pair[index], proportion=False
            )

            if cur_cover_length > 0:
                gold_elem_length += cur_gold_length
                cover_elem_length += cur_cover_length

            else:
                return False, 0

        if polarity and gold_tuple_pair[-1] != predict_tuple_pair[-1]:
            return False, 0

        return True, cover_elem_length / gold_elem_length if gold_elem_length != 0 else 0

    ####################################################################################################################
    # Evaluation Write Result txt Part.
    ####################################################################################################################
    def print_measure(self, measure, measure_file, measure_type="exact"):
        """
        :param measure: {elem: {P: xx, R: xx, F: xx}.....}
        :param measure_file: a file path to write.
        :param measure_type: "exact", "binary" or "prop"
        :return:
        """
        assert measure_type in {"exact", "prop", "binary"}, "unknown measure type."

        with open(measure_file, "a") as f:
            self.standard_print(measure, measure_type, f)

    def best_model(self, measure_file):
        """
        :param measure_file:
        :return:
        """
        with open(measure_file, "a") as f:
            print("========================================", file=f)
            print("fold: {} Best Model Measure".format(self.fold), file=f)
            print("========================================", file=f)

            self.standard_print(self.optimize_exact_measure, "Exact", f)
            self.standard_print(self.optimize_prop_measure, "Proportional", f)
            self.standard_print(self.optimize_binary_measure, "Binary", f)

            print("========================================", file=f)

    @staticmethod
    def standard_print(measure, measure_type, file_point):
        print("========================================", file=file_point)
        print("{} Measure".format(measure_type), file=file_point)
        print("========================================", file=file_point)
        for elem in measure.keys():
            if elem == "sent_acc":
                print("Comparative Sentence Label Accuracy is {:.2f}%".format(measure[elem]['F']), file=file_point)
                continue

            if elem == "elem_acc":
                print("Predicate Correspond Sentence Label Accuracy is {:.2f}%".format(measure[elem]['F']),
                      file=file_point)
                continue

            if elem == "polarity_acc":
                print("Polarity Label Accuracy is {:.2f}%".format(measure[elem]['F']),
                      file=file_point)
                continue

            print("{} Measure's {} Precision value is {:.2f}"
                  .format(measure_type, elem, measure[elem]['P']), file=file_point)

            print("{} Measure's {} Recall value is {:.2f}"
                  .format(measure_type, elem, measure[elem]['R']), file=file_point)

            print("{} Measure's {} F-Measure value is {:.2f}"
                  .format(measure_type, elem, measure[elem]['F']), file=file_point)

    def elem_dict_to_string(self, token_list, data_dict):
        """
        :param token_list:
        :param data_dict: {elem: [(s_index, e_index)]}
        :return:
        """
        elem_str = "["

        # cur_elem_col = ["subject", "object", "aspect", "result"]
        cur_elem_col = self.elem_col
        for index, elem in enumerate(cur_elem_col):
            elem_str += "["
            for elem_index, elem_representation in enumerate(data_dict[elem]):
                if len(elem_representation) == 3:
                    s_index, e_index, polarity = elem_representation
                else:
                    polarity = None
                    s_index, e_index = elem_representation

                for k in range(s_index, e_index):
                    elem_str += str(k) + "_" + token_list[k]
                    elem_str += " " if k != e_index - 1 else ""

                if polarity is not None:
                    elem_str += "..."
                    elem_str += self.config.val.polarity_col[int(polarity) + 1]

                elem_str += " , " if elem_index != len(data_dict[elem]) - 1 else "]"

            elem_str += "]" if len(data_dict[elem]) == 0 else ""
            elem_str += "; " if index != len(self.elem_col) - 1 else ""

        elem_str += "]"

        return elem_str

    ####################################################################################################################
    # Average Measure Process Part
    ####################################################################################################################
    def avg_model(self, write_file):
        """
        :param write_file:
        :return:
        """
        with open(write_file, "a") as f:
            print("========================================", file=f)
            print("fold: {} Best Model Measure".format(self.fold), file=f)
            print("========================================", file=f)

            self.standard_print(self.avg_exact_measure, "Exact", f)
            self.standard_print(self.avg_prop_measure, "Proportional", f)
            self.standard_print(self.avg_binary_measure, "Binary", f)

            print("========================================", file=f)

    @staticmethod
    def add_fold_measure(avg_measure, opt_measure, fold_num=5):
        """
        :param avg_measure: store all fold average measure.
        :param opt_measure: each fold's optimize measure.
        :param fold_num: fold number.
        :return:
        """
        # logger.info("average mesurer is: {}".format(avg_measure))
        logger.info("optimizer mesurer is: {}".format(opt_measure))

        if len(avg_measure) == 0:
            avg_measure = copy.deepcopy(opt_measure)

            for elem, eval_dict in opt_measure.items():
                if not isinstance(eval_dict, dict):
                    continue

                for eval_key, num in eval_dict.items():
                    avg_measure[elem][eval_key] = num / fold_num
        else:
            for elem, eval_dict in opt_measure.items():
                if not isinstance(eval_dict, dict):
                    continue

                for eval_key, num in eval_dict.items():
                    avg_measure[elem][eval_key] += num / fold_num

        return avg_measure

    def store_result_to_csv(self, model_name, csv_file_path):
        """
        :param model_name: [model name]
        :param csv_file_path: write csv file path.
        :return:
        """
        row_label = ["measure_type"]
        data = [["Exact"], ["Prop"], ["Binary"]]

        for elem in self.avg_exact_measure.keys():
            row_label.append(elem)
            data[0].append(round(self.avg_exact_measure[elem]['F'], 2))
            data[1].append(round(self.avg_prop_measure[elem]['F'], 2))
            data[2].append(round(self.avg_binary_measure[elem]['F'], 2))

        with open(csv_file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(model_name)
            writer.writerow(row_label)
            writer.writerows(data)

## Đánh giá độ chính xác khi trích xuất các element
class ElementEvaluation(BaseEvaluation):
    def __init__(self, config, target=None, attn_mask=None, elem_col=None, ids_to_tags=None, fold=0,
                 save_model=False, comparative_identity=False, gold_sent_label=None):
        super(ElementEvaluation, self).__init__(
            config, elem_col=elem_col, ids_to_tags=ids_to_tags, fold=fold, save_model=save_model
        )
        self.comparative_identity = comparative_identity
        self.gold_sent_label = gold_sent_label

        ## trường hợp đánh giá hiệu suất mô hình training
        if attn_mask is not None and target is not None:
            elem_label_ids, result_label_ids = target

            predicate_target, elem_target = [], []
            for index in range(len(result_label_ids)):
                predicate_target.append(result_label_ids[index][attn_mask[index] == 1].tolist())

                cur_elem_label = []
                for k in range(len(elem_label_ids[index])):
                    cur_elem_label.append(elem_label_ids[index][k][attn_mask[index] == 1].tolist())
                elem_target.append(cur_elem_label)

            self.gold_dict = self.get_elem_dict(target)

        ## trường hợp trích xuất các element
        else:
            self.gold_dict = None
        # store predict if comparative sentence.
        self.predict_sent_label = []

    def eval_model(self, measure_file, model=None, model_path=None, multi_elem_score=True):
        """
        :param measure_file: a file path to write result.
        :param model: use to save.
        :param model_path: a file path to save model.
        :param input_ids: process on chinese version token.
        :param multi_elem_score: denote if calculate macro and micro.
        :return:
        """
        assert (self.save_model and model is not None) or not self.save_model

        # using predicted output to get elem_dict.
        self.predict_dict = self.get_elem_dict((self.elem_hat, self.result_hat))
        # for i in range(5):
        #     logger.info("Prediction in eval pharase: {}".format(self.predict_dict[i]))

        # eval part do not drop elem.
        # Nếu không phải câu comparative (predict_sent_label == 0) thì chuyển predict_dict thành []
        if self.comparative_identity:
            self.predict_dict = self.mask_non_comparative(self.predict_dict, self.predict_sent_label)

        # using exact, proportional and binary measure model.
        key_col = self.elem_col

        gold_num = self.init_elem(key_col)
        predict_num = self.init_elem(key_col)

        exact_correct_num, prop_correct_num, binary_correct_num = \
            self.init_elem(key_col), self.init_elem(key_col), self.init_elem(key_col)

        assert len(self.predict_dict) == len(self.gold_dict)

        # for i in range(3):
        #     logger.info("Predict dict in eval phrase (after mask non-com): {}".format(self.predict_dict[i]))
        #     logger.info("Gold dict in eval phrase: {}".format(self.gold_dict[i]))

        # calculate elem dict.
        for index in range(len(self.gold_dict)):
            # sequence elem dict: {elem: {s_index, e_index}}
            gold_sequence_elem_dict = self.gold_dict[index]
            predict_sequence_elem_dict = self.predict_dict[index]

            # print(gold_sequence_elem_dict)
            # print(predict_sequence_elem_dict)

            for elem in self.elem_col:
                gold_num[elem] += len(gold_sequence_elem_dict[elem])
                predict_num[elem] += len(predict_sequence_elem_dict[elem])

                ## cur_exact_num = [correct_num, result_polarity_correct_num, polaruty_correct_num_col]
                # tính số lượng dự đoán các element
                ## cur_prop_num, cur_binary_num =[correct_num, result_polarity_correct_num, polaruty_correct_num_col]
                # tính tỉ lệ cover = chiều dài chuỗi dự đoán đúng / chiều dài chuỗi thực tế
                # binary ~ exatract, prop: cover_rate
                cur_exact_num, cur_prop_num, cur_binary_num = self.get_elem_num(
                    gold_sequence_elem_dict[elem], predict_sequence_elem_dict[elem]
                )

                assert cur_binary_num[0] <= len(gold_sequence_elem_dict[elem]), "[ERROR] eval error!"
                assert cur_binary_num[0] <= len(predict_sequence_elem_dict[elem]), "[ERROR] eval error!"

                # print(elem, cur_exact_num[0], cur_prop_num[0], cur_binary_num[0])
                exact_correct_num[elem] += cur_exact_num[0]
                prop_correct_num[elem] += cur_prop_num[0]
                binary_correct_num[elem] += cur_binary_num[0]

        # print(gold_num)
        # logger.info("Number of gold ")
        # calculate f-score.
        exact_measure = self.get_f_score(gold_num, predict_num, exact_correct_num, multi_elem_score)
        prop_measure = self.get_f_score(gold_num, predict_num, prop_correct_num, multi_elem_score)
        binary_measure = self.get_f_score(gold_num, predict_num, binary_correct_num, multi_elem_score)

        # add sentence identify accuracy
        if self.gold_sent_label is not None:
            exact_measure = self.measure_add_accuracy(
                exact_measure, self.predict_sent_label, self.predict_dict
            )
            prop_measure = self.measure_add_accuracy(
                prop_measure, self.predict_sent_label, self.predict_dict
            )
            binary_measure = self.measure_add_accuracy(
                binary_measure, self.predict_sent_label, self.predict_dict
            )

        # print result in file
        logger.info("Extraction measure: {}".format(exact_measure))
        logger.info("Binary measure: {}".format(binary_measure))
        logger.info("Proportion measure: {}".format(prop_measure))

        self.print_measure(exact_measure, measure_file, measure_type='exact')
        self.print_measure(prop_measure, measure_file, measure_type='prop')
        self.print_measure(binary_measure, measure_file, measure_type='binary')

        optimize_measure_type = "micro"

        if self.is_optimize(self.optimize_exact_measure, exact_measure, optimize_measure_type):
            self.optimize_exact_measure = copy.deepcopy(exact_measure)
            self.optimize_prop_measure = copy.deepcopy(prop_measure)
            self.optimize_binary_measure = copy.deepcopy(binary_measure)

            if self.save_model:
                torch.save(model, model_path)

        # init predict output.
        self.result_hat, self.elem_hat = [], []
        self.predict_sent_label = []

    def add_sent_label(self, output):
        """
        :param output: a tensor, shape is [batch_size, seq_length]
        """
        for index in range(output.size(0)):
            if self.config.device == "cpu":
                mask_output = output[index].numpy().tolist()
            else:
                mask_output = output[index].cpu().numpy().tolist()

            self.predict_sent_label.append(mask_output)

    @staticmethod
    ##
    def mask_non_comparative(data_dict, predict_label):
        """
        :param data_dict:
        :param predict_label:
        :return:
        """
        assert len(data_dict) == len(predict_label), "data length error!"

        for index in range(len(data_dict)):
            if predict_label[index] == 0:
                for elem in data_dict[index].keys():
                    data_dict[index][elem] = []

        return data_dict

    @staticmethod
    def count_polarity_num(result_elem_col):
        """
        :param result_elem_col: [(s_index, e_index, polarity)]
        :return:
        """
        polarity_num = [0] * 4

        for index in range(len(result_elem_col)):
            polarity_num[result_elem_col[index][-1] + 1] += 1

        return polarity_num

    ####################################################################################################################
    # sentence identify accuracy
    ####################################################################################################################

    def get_sent_identify_accuracy(self, y_hat):
        """
        :param y_hat: [n]
        :return:
        """
        assert len(self.gold_sent_label) == len(y_hat), "sentence label length error!"

        gold, predict = np.array(self.gold_sent_label).reshape(-1), np.array(y_hat)

        return np.sum(gold == predict) / len(gold) * 100

    @staticmethod
    def is_null_elem_dict(data_dict):
        """
        :param data_dict:
        :return:
        """
        for key, value in data_dict.items():
            if len(value) > 0:
                return True

        return False

    def get_predicate_identify_accuracy(self, elem_dict):
        """
        :param elem_dict: [n, elem_dict], elem_dict: {elem: (s_index, e_index)}
        :return:
        """
        predicate_predict = []
        for index in range(len(elem_dict)):
            if self.is_null_elem_dict(elem_dict[index]):
                predicate_predict.append(1)
            else:
                predicate_predict.append(0)

        gold = np.array(self.gold_sent_label).reshape(-1)
        predicate_predict = np.array(predicate_predict)
        return np.sum(gold == predicate_predict) / len(gold) * 100

    def measure_add_accuracy(self, measure, y_hat, elem_dict):
        """
        :param measure:
        :param y_hat:
        :param elem_dict:
        :return:
        """
        measure_val = {"P", "R", "F"}

        for val in measure_val:
            if self.config.model_type not in {"crf"}:
                if "sent_acc" not in measure:
                    measure['sent_acc'] = {}
                measure['sent_acc'][val] = self.get_sent_identify_accuracy(y_hat)

            if self.config.model_type not in {"classification"}:
                if "elem_acc" not in measure:
                    measure['elem_acc'] = {}
                measure['elem_acc'][val] = self.get_predicate_identify_accuracy(elem_dict)

        return measure

    def print_elem_result(self, input_ids, mask, write_path, drop_span=True):
        """
        :param input_ids: [n, sequence_length]
        :param mask:
        :param write_path:
        :param drop_span:
        :return:
        """
        assert len(input_ids) == len(self.gold_dict), "input_ids appear error"

        write_str, elem_str = "", ""
        for index in range(len(input_ids)):
            if drop_span: ## drop [cls], [sep]
                token_list = eval_shared_modules.bert_data_transfer(
                    self.bert_tokenizer, input_ids[index][mask[index] == 1][1:-1], data_type="ids"
                )

            else:
                token_list = eval_shared_modules.bert_data_transfer(
                    self.bert_tokenizer, input_ids[index][mask[index] == 1], data_type="ids"
                )

            write_str += " ".join(token_list) + "\n"

            write_str += self.elem_dict_to_string(token_list, self.predict_dict[index]) + "\n"
            write_str += self.elem_dict_to_string(token_list, self.gold_dict[index]) + "\n"
            write_str += str(self.predict_dict[index]) + "\n"
            

            # if index < 3:
            #     logger.info("Gold sentence: {}".format(self.bert_tokenizer.decode(input_ids[index][1:-1], skip_special_tokens = True)))
            #     logger.info("Gold token list: {}".format(self.gold_dict[index]))
            #     logger.info("Predict dict: {} ".format(self.predict_dict[index]))

        with open(write_path, "w", encoding='utf-8', errors='ignore') as f:
            f.write(write_str)

    ####################################################################################################################
    # Generate make pair model data
    ####################################################################################################################
    def generate_elem_representation(self, gold_pair_label, feature_embed, bert_feature_embed, feature_type=0):
        """
        :param gold_pair_label: [(s_index, e_index)] * 5 = data_dict['tuple_pair_col]
        :param feature_embed: [N, 3, sequence_length, feature_dim], feature_dim=5?
        :param bert_feature_embed:
        :param feature_type：
        :return:
        """
        # a list elem_dict: {elem: [(s_index, e_index)]}
        self.predict_dict = self.get_elem_dict((self.elem_hat, self.result_hat))

        # logger.info("Number of prediction in first stage test: {}".format(len(self.predict_dict)))

        candidate_pair_col = []

        # elem_col = {"entity_1", "entity_2", "aspect", "result"}
        for index in range(len(self.predict_dict)):
            cur_candidate_pair_col = []
            cur_predict_elem_dict = self.predict_dict[index]

            for elem in self.elem_col:
                if len(cur_predict_elem_dict[elem]) != 0:
                    cur_elem = cur_predict_elem_dict[elem]
                else:
                    cur_elem = [(-1, -1)]

                cur_candidate_pair_col = eval_shared_modules.cartesian_product(cur_candidate_pair_col, cur_elem)

            candidate_pair_col.append(cur_candidate_pair_col)

        logger.info("number candicate quintupe after castesion: {}".format(len(candidate_pair_col)))
        # for i in range(5):
        #     logger.info("Gold pair {} in test phrase: {}".format(i, gold_pair_label[i]))
        #     logger.info("Predicted tuple {} in test phrase: {}".format(i, self.predict_dict[i]))
        #     logger.info("Candidate tuple  {} in test pharase: {}".format(i, candidate_pair_col[i]))


        pair_representation = self.create_pair_representation(
            candidate_pair_col, feature_embed, bert_feature_embed, feature_type=feature_type
        )

        make_pair_label = self.create_pair_label(candidate_pair_col, gold_pair_label)

        self.elem_hat, self.result_hat = [], []

        return candidate_pair_col, pair_representation, make_pair_label, feature_embed, bert_feature_embed

    def create_pair_representation(self, candidate_col, feature_out, bert_feature_out, feature_type=0):
        """
        :param candidate_col: [n, tuple_pair_num, tuple_pair], tuple_pair: [(s_index, e_index)]
        :param feature_out: [n, 4, sequence_length, feature_dim]
        :param bert_feature_out: [n, sequence_length, feature_dim]
        :param feature_type: 0 表示 5维 + 768维，1 表示 5维，2 表示 768维
        :return:
        """
        pair_input, hidden_size = [], 5
        if "bert" in self.config.model_mode :
            encode_hidden_size = 768
        else:
            encode_hidden_size = self.config.hidden_size * 2

        for index in range(len(candidate_col)):
            pair_representation = []
            for pair_index in range(len(candidate_col[index])):
                each_pair_representation = []

                # add polarity maybe need change###########

                # skip polarity
                for elem_index in range(4):
                    s, e = candidate_col[index][pair_index][elem_index]
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
                        # tensor có kích thước = 5 lưu giá trị trung bình biểu diễn của các element (từ s_index -> e_index), đại diện cho 4 element và 1 comp label
                        # concatenate element extraction from s-> e_index with hidden_representation of all sentence but from (s->e_index of element) 
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

                if self.config == "cuda":
                    cur_representation = torch.cat(each_pair_representation, dim=-1).view(-1).cpu().numpy().tolist()
                else:
                    cur_representation = torch.cat(each_pair_representation, dim=-1).view(-1).numpy().tolist()

                pair_representation.append(cur_representation)

            # pair_representation = torch.stack(pair_representation, dim=0)

            pair_input.append(pair_representation)

        return pair_input

    @staticmethod
    def is_equal_tuple_pair(candidate_tuple_col, truth_tuple_col, null_pair):
        if truth_tuple_col == null_pair:
            return False

        if len(candidate_tuple_col) != len(truth_tuple_col):
            if candidate_tuple_col == truth_tuple_col[:-1]:
                return True
            return False
        else:
            if candidate_tuple_col == truth_tuple_col:
                return True
            return False

    def create_pair_label(self, candidate_col, truth_pair_label):
        """
        :param candidate_col: shape is [n, tuple_pair_num, tuple_pair]
        :param truth_pair_label: shape is [n, tuple_pair_num, tuple_pair]
        :return:
        """
        pair_label_col, null_pair = [], [(-1, -1)] * 5
        for i in range(len(candidate_col)):
            # cartesian product pair num
            is_pair_label = []
            for j in range(len(candidate_col[i])):
                # truth predicate pair num
                isExist = False
                for k in range(len(truth_pair_label[i])): # check if  candidate pair is match with truth pair
                    if self.is_equal_tuple_pair(candidate_col[i][j], truth_pair_label[i][k], null_pair):
                        isExist = True

                is_pair_label.append(1 if isExist else 0)

            pair_label_col.append(is_pair_label)

        return pair_label_col


class PairEvaluation(BaseEvaluation):
    def __init__(self, config, candidate_pair_col, gold_pair_col, elem_col, ids_to_tags, save_model=False, fold=0, token_col=None):
        super(PairEvaluation, self).__init__(
            config, elem_col=elem_col, ids_to_tags=ids_to_tags, save_model=save_model, fold=fold
        )

        self.candidate_pair_col = candidate_pair_col
        self.gold_pair_col = gold_pair_col

        self.y_hat = []
        self.polarity_hat = []
        if token_col:
            self.standard_token_col, self.bert_token_col = token_col
        else:
            self.standard_token_col, self.bert_token_col = [], []

    def eval_model(self, measure_file, model=None, model_path=None, polarity=False, initialize=(False, False)):
        """
        :param measure_file: a file path to write result.
        :param model: use to save.
        :param model_path: a file path to save model.
        :param polarity:
        :param initialize: (polarity, pair)
        :return:
        """
        assert (self.save_model and model is not None) or not self.save_model

        # using exact, proportional and binary measure model.
        predict_num, gold_num = {"init_pair": 0.0, "pair": 0.0}, {"init_pair": 0.0, "pair": 0.0}
        exact_correct_num, prop_correct_num = {"init_pair": 0.0, "pair": 0.0}, {"init_pair": 0.0, "pair": 0.0}
        binary_correct_num = {"init_pair": 0.0, "pair": 0.0}

        predict_tuple_pair_col = self.get_predict_truth_tuple_pair(self.candidate_pair_col)
        print(len(self.gold_pair_col), len(predict_tuple_pair_col))

        assert len(self.gold_pair_col) == len(predict_tuple_pair_col), "data length error!"

        # for i in range(5):
        #     logger.info("Gold pair label: {}".format(self.gold_pair_col[i]))
        #     logger.info("Candidate pair label: {}".format(self.candidate_pair_col[i]))
        #     logger.info("Predict pair label: {}".format(predict_tuple_pair_col[i]))

        # calculate elem dict.
        # tuple_str = ""
        for index in range(len(self.gold_pair_col)):
            gold_sequence_pair_col = self.gold_pair_col[index]
            predict_sequence_pair_col = predict_tuple_pair_col[index]

            gold_num['pair'] += self.get_effective_pair_num(gold_sequence_pair_col)
            predict_num['pair'] += self.get_effective_pair_num(predict_sequence_pair_col)

            gold_num['init_pair'] += self.get_effective_pair_num(gold_sequence_pair_col)
            predict_num['init_pair'] += self.get_effective_pair_num(self.candidate_pair_col[index])

            # count số lượng tuple predict đúng
            cur_exact_num, cur_prop_num, cur_binary_num = self.get_pair_num(
                gold_sequence_pair_col, predict_sequence_pair_col, polarity=polarity
            )

            cur_fake_exact_num, cur_fake_prop_num, cur_fake_binary_num = self.get_pair_num(
                gold_sequence_pair_col, self.candidate_pair_col[index], polarity=False
            )

            assert cur_exact_num <= cur_prop_num <= cur_binary_num, "eval calculate error!"
            assert cur_fake_exact_num <= cur_fake_prop_num <= cur_fake_binary_num, "eval calculate error!"

            # tuple_str += self.print_tuple_pair(
            #     gold_sequence_pair_col, predict_sequence_pair_col, [cur_exact_num, cur_binary_num]
            # )

            exact_correct_num['pair'] += cur_exact_num
            prop_correct_num['pair'] += cur_prop_num
            binary_correct_num['pair'] += cur_binary_num

            exact_correct_num['init_pair'] += cur_fake_exact_num
            prop_correct_num['init_pair'] += cur_fake_prop_num
            binary_correct_num['init_pair'] += cur_fake_binary_num

        # with open("./tuple_pair_output.txt", "w", encoding='utf-8') as f:
        #     f.write(tuple_str)

        logger.info("Gold num: {}".format(gold_num))
        logger.info("Predict num: {}".format(predict_num))

        # calculate f-score.
        exact_measure = self.get_f_score(gold_num, predict_num, exact_correct_num, multi_elem_score=False)
        prop_measure = self.get_f_score(gold_num, predict_num, prop_correct_num, multi_elem_score=False)
        binary_measure = self.get_f_score(gold_num, predict_num, binary_correct_num, multi_elem_score=False)

        # add polarity accuracy.
        exact_measure = self.get_polarity_acc(exact_measure, exact_correct_num['pair'], gold_num['pair'])
        prop_measure = self.get_polarity_acc(prop_measure, prop_correct_num['pair'], gold_num['pair'])
        binary_measure = self.get_polarity_acc(binary_measure, binary_correct_num['pair'], gold_num['pair'])

        # if predict_num['init_pair'] == 0:
        #     keep_rate = 0.0
        # else:
        keep_rate = predict_num['pair'] / predict_num['init_pair'] * 100
        keep_rate_dict = {"P": keep_rate, "R": keep_rate, "F": keep_rate}
        exact_measure['keep_rate'], prop_measure['keep_rate'] = keep_rate_dict, keep_rate_dict
        binary_measure['keep_rate'] = keep_rate_dict

        logger.info("Extraction result: {}".format(exact_measure))
        logger.info("Proportion result: {}".format(prop_measure))
        logger.info("Binary result: {}".format(binary_measure))

        # print result in file
        self.print_measure(exact_measure, measure_file, measure_type='exact')
        self.print_measure(prop_measure, measure_file, measure_type='prop')
        self.print_measure(binary_measure, measure_file, measure_type='binary')

        if self.is_optimize(self.optimize_exact_measure, exact_measure, "pair"):
            self.optimize_exact_measure = copy.deepcopy(exact_measure)
            self.optimize_prop_measure = copy.deepcopy(prop_measure)
            self.optimize_binary_measure = copy.deepcopy(binary_measure)

            if self.save_model:
                torch.save(model, model_path)
        
        if polarity:
            with open("./ModelResult/" + "/test_quintuple_wi_polarity_result_file.txt", "w", encoding="utf8") as fout:
                for index, pair in enumerate(predict_tuple_pair_col):
                    fout.write(f"{pair}\t{self.y_hat[index]}\t{self.polarity_hat[index]}\n")
            fout.close()
        else:
            with open("./ModelResult/" + "/test_quintuple_wo_polarity_result_file.txt", "w", encoding="utf8") as fout:
                for index, pair in enumerate(predict_tuple_pair_col):
                    fout.write(f"{pair}\t{self.y_hat[index]}\n")
            fout.close()

        if initialize[0]:
            self.polarity_hat = [] # predict comparative label of sentence
        elif initialize[1]:
            self.y_hat = [] #identify whether sentence is comparative

    @staticmethod
    def get_effective_pair_num(tuple_pair_col):
        """
        :param tuple_pair_col:
        :return:
        """
        elem_length = len(tuple_pair_col[0]) if len(tuple_pair_col) != 0 else 5
        null_pair, pair_num = [(-1, -1)] * elem_length, 0
        for index in range(len(tuple_pair_col)):
            if tuple_pair_col[index] == null_pair:
                continue
            pair_num += 1
        return pair_num

    def add_pair_data(self, match_label):
        if self.config.device == "cuda":
            match_label = match_label.cpu().numpy().tolist()
        else:
            match_label = match_label.numpy().tolist()

        self.y_hat.append(match_label)

    def add_polarity_data(self, predict_polarity):
        if self.config.device == "cuda":
            predict_polarity = predict_polarity.cpu().numpy().tolist()
        else:
            predict_polarity = predict_polarity.numpy().tolist()

        self.polarity_hat.append(predict_polarity)

    @staticmethod
    def get_polarity_acc(measure, correct_num, gold_num):
        """
        :param measure:
        :param correct_num:
        :param gold_num:
        :return:
        """
        if gold_num == 0:
            acc = 0
        else:
            acc = correct_num / gold_num * 100

        measure['polarity_acc'] = {}
        for val in {"P", "R", "F"}:
            measure['polarity_acc'][val] = acc

        return measure

    def get_predict_truth_tuple_pair(self, candidate_tuple_pair_col):
        """
        :param candidate_tuple_pair_col:
        :return:
        """
        truth_tuple_pair_col = []

        # with polarity and is_pair. (in training)
        if len(self.y_hat) != 0 and len(self.polarity_hat) != 0:
            logger.info("Len(y_hat) and len(polarity_hat) is not 0")

            for index in range(len(candidate_tuple_pair_col)):
                cur_predicate_tuple_pair = []

                # Thêm polarity vào các candidate có y_hat = 1 (là comparative sentence).
                for k in range(len(self.y_hat[index])):
                    if self.y_hat[index][k] == 1:
                        cur_predicate_tuple_pair.append(
                            self.add_polarity_to_tuple_pair(candidate_tuple_pair_col[index][k], self.polarity_hat[index][k])
                        )
                    else:
                        cur_predicate_tuple_pair.append([(-1, -1)]*5)

                truth_tuple_pair_col.append(cur_predicate_tuple_pair)

        elif len(self.polarity_hat) != 0: # test stage 3
            logger.info("len(polarity_hat) is not 0")
            for index in range(len(candidate_tuple_pair_col)):
                cur_predicate_tuple_pair = []

                # Thêm polarity vào các candidate
                for k in range(len(self.polarity_hat[index])):
                    if len(candidate_tuple_pair_col[index][k]) == 0:
                        cur_predicate_tuple_pair = [(-1, -1)]*4
                    cur_predicate_tuple_pair.append(
                        self.add_polarity_to_tuple_pair(candidate_tuple_pair_col[index][k], self.polarity_hat[index][k])
                    )

                truth_tuple_pair_col.append(cur_predicate_tuple_pair)

        elif len(self.y_hat) != 0: # test stage 2
            logger.info("Len(y_hat) is not 0")
            for index in range(len(candidate_tuple_pair_col)):
                cur_predicate_tuple_pair = []

                # drop none-pair and add polarity to pair.
                for k in range(len(self.y_hat[index])):
                    if self.y_hat[index][k] == 1: # comparative sentence
                        cur_predicate_tuple_pair.append(copy.deepcopy(candidate_tuple_pair_col[index][k]))
                    else:
                        cur_predicate_tuple_pair.append([(-1, -1)]*5)

                truth_tuple_pair_col.append(cur_predicate_tuple_pair)

        assert len(self.y_hat) != 0 or len(self.polarity_hat) != 0, "[ERROR] Data Process Error!"

       
        return truth_tuple_pair_col

    @staticmethod
    def add_polarity_to_tuple_pair(tuple_pair, polarity):
        return copy.deepcopy(tuple_pair + [(int(polarity - 1), int(polarity - 1))])

    # def print_tuple_pair(self, gold_token_col, model_path):
    #     """
    #     :param gold_tuple_pair:
    #     :param predict_tuple_pair:
    #     :param correct_num:
    #     :return:
    #     """
    #     write_str = ""
    #     for index in range(len(gold_tuple_pair)):
    #         write_str += self.tuple_pair_to_string(gold_tuple_pair[index], gold_token_col[index])

    #     write_str += "----------------------------------\n"

    #     for index in range(len(predict_tuple_pair)):
    #         write_str += self.tuple_pair_to_string(predict_tuple_pair[index], gold_token_col[index])

    #     for index in range(len(correct_num)):
    #         write_str += str(correct_num[index])

    #         if index != len(correct_num) - 1:
    #             write_str += " "
    #         else:
    #             write_str += "\n"

    
    def tuple_pair_to_string(tuple_pair, gold_token_col):
        """
        :param tuple_pair:
        :return:
        """
        write_str = "["
        elem_col = {0: "subject", 1: "object", 2: "aspect", 3:"predicate", 4: "label"}
        for index, pair in enumerate(tuple_pair):

            write_str += "(" + str(tuple_pair[index][0]) + ", " + str(tuple_pair[index][1]) + ")"

            if index != len(tuple_pair) - 1:
                write_str += " , "
            else:
                write_str += "]\n"

        return write_str

import torch
import numpy as np
import random
import os, sys
import argparse
import Config
import logging
# logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
logger = logging.getLogger()

from data_utils import shared_utils, create_dataset, data_loader_utils
from eval_utils import create_eval, eval_shared_modules
from eval_utils.base_eval import BaseEvaluation, ElementEvaluation, PairEvaluation
import train_test_utils


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.bendmark = False
    torch.nn.Module.dump_patches = True

def prepare_logger(args):
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    formatter = logFormatter = logging.Formatter(fmt='[%(asctime)s - %(name)s:%(lineno)d]: %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
    log_file = os.path.join(os.getcwd(), "run.log")
    file_handler = logging.FileHandler(log_file, mode="w", encoding=None, delay=False)
    console_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.handlers = [console_handler, file_handler]


def TerminalParser():
    parser = argparse.ArgumentParser()
    parser.description = 'choose train data and test data file path'

    parser.add_argument('--seed', help='random seed', type = int, default=42)
    parser.add_argument('--batch', help='input data batch size', type= int, default=16)
    parser.add_argument('--epoch', help='number of run times', type=int, default=20)
    parser.add_argument('--second_epoch', help='number of second times', type=int, default=50)
    parser.add_argument('--fold', help='the fold of data', type= int, default=5)

    parser.add_argument('--input_size', help='the size of encoder embedding', type=int, default=512)
    parser.add_argument('--hidden_size', help='the size of hidden embedding', type=int, default=512)
    parser.add_argument('--num_layers', help='the number of layers', type=int, default=2)

    parser.add_argument('--model_mode', help='bert or norm', type=str, default='bert') #"bert-base-multilingual-cased"
    parser.add_argument('--program_mode', help='run or test', default='run')
    parser.add_argument('--stage_model', help='first or second', default='first')
    parser.add_argument('--model_type', help='bert_crf, bert_crf_mtl', default='crf')
    parser.add_argument('--position_sys', help='BIES or BI or SPAN', default='BMES')

    parser.add_argument('--device', help='run program in device type', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--file_type', help='the type of data set', default='data')
    # parser.add_argument('--premodel_path', help='the type of pretrained model', default= "bert-base-multilingual-cased")
    parser.add_argument('--embed_dropout', help='probability of embedding dropout', type=int, default=0.1)
    parser.add_argument('--factor', help='the trade-off hyperparameter in class-weight entropy loss cal', type=float, default=0.4)
    parser.add_argument('--bert_lr', help='learning rate of bert model', type=float, default=2e-5)
    parser.add_argument('--linear_lr', help='the learning rate of linear layer', type=float, default=2e-5)
    parser.add_argument('--crf_lr', help='the learning rate of crf layers', type=float, default=0.01)

    args = parser.parse_args()
    return args

def get_necessary_parameters(args):
    param_dict = {"file_type": args.file_type,
                "model_mode": args.model_mode,
                "stage_type": args.model_type,
                "model_type": args.model_type,
                # "epoch": args.epoch,
                # "batch_size": args.batch,
                "program_mode": args.program_mode}
    return param_dict

def main():

    args = TerminalParser()
    prepare_logger(args)
    set_seed(args.seed)

    config = Config.BaseConfig(args)
    config_parameters = get_necessary_parameters(args)

    if args.stage_model == 'first':
        model_parameters = {"embed_dropout": args.embed_dropout}
    else:
        model_parameters = {"embed_dropout": args.embed_dropout, "factor": args.factor}
    
    optimizer_parameters = None

    model_name  = shared_utils.parameters_to_model_name(
        {"config": config_parameters, "model": model_parameters}
    )
    logger.info("Using model: {}".format(model_name) )

    dataset = create_dataset.Dataset(config)
    dataset.generate_data()

    global_eval = BaseEvaluation(config, config.val.elem_col)
    global_pair_eval = BaseEvaluation(config, config.val.elem_col)

    logger.info("=======================CREATE DATA LOADER==========================")
    logger.info("Train dataset input ids shape: {}".format(dataset.train_data_dict['input_ids'].shape) )

    train_loader = data_loader_utils.create_first_data_loader(dataset.train_data_dict, config.batch_size, "train")
    dev_loader = data_loader_utils.create_first_data_loader(dataset.dev_data_dict, config.batch_size, "dev")
    test_loader = data_loader_utils.create_first_data_loader(dataset.test_data_dict, config.batch_size, "test")
    # logger.info("First object in data loader is {}".format(next()) )

    if config.stage_model == 'first' and config.program_mode != 'test':
        logger.info("%"*30+"FIRST STAGE RUNNING" + "%"*30)
        first_data_loader = [train_loader, dev_loader, test_loader]
        dev_comp_eval = create_eval.create_first_stage_eval(
            config,
            (dataset.dev_data_dict['multi_label'], dataset.dev_data_dict['result_label']),
            dataset.dev_data_dict['comparative_label'],
            dataset.dev_data_dict['attn_mask'],
            save_model=True
        )

        test_comp_eval = create_eval.create_first_stage_eval(
            config,
            (dataset.test_data_dict['multi_label'], dataset.test_data_dict['result_label']),
            dataset.test_data_dict['comparative_label'],
            dataset.test_data_dict['attn_mask'],
            save_model=False
        )

        comp_eval = [dev_comp_eval, test_comp_eval, global_eval]

        train_test_utils.first_stage_model_main(
            config, dataset, first_data_loader, comp_eval,
            model_parameters, optimizer_parameters,
            model_name
        )
    
    elif config.program_mode == 'test' and config.stage_model == 'first':
        dev_parameters = ["./ModelResult/" + model_name + "/dev_elem_result.txt",
                          "./PreTrainModel/" + model_name + "/dev_model"]
        logger.info("%"*30+"FIRST STAGE TESTING" + "%"*30)
        # logger.info("=====================TEST FIRST STAGE====================")
        predicate_model = torch.load(dev_parameters[1])

        test_parameters = ["./ModelResult/" + model_name + "/test_elem_result.txt", None]

        test_comp_eval = create_eval.create_first_stage_eval(
            config,
            (dataset.test_data_dict['multi_label'], dataset.test_data_dict['result_label']),
            dataset.test_data_dict['comparative_label'],
            dataset.test_data_dict['attn_mask'],
            save_model=False
        )

        train_test_utils.first_stage_model_test(
            predicate_model, config, test_loader, test_comp_eval, test_parameters
        )

        test_comp_eval.print_elem_result(
            dataset.test_data_dict['input_ids'], dataset.test_data_dict['attn_mask'],
            "./ModelResult/" + model_name + "/test_result_file" + ".txt", drop_span=False
        )

        # add average measure.
        eval_shared_modules.calculate_average_measure(test_comp_eval, global_eval)

    elif config.program_mode == "test" and config.stage_model == "second":
        # 0: 768 + 5, 1: 5, 2: 768
        logger.info("%"*30+"SECOND STAGE TESTING" + "%"*30)
        feature_type = 0

        # using evaluation to generate index col and pair label.
        generate_second_res_eval = ElementEvaluation(
            config, elem_col=config.val.elem_col,
            ids_to_tags=config.val.invert_norm_id_map
        )

        if model_name.find("ele") != -1:
            cross_model_name = model_name.replace("ele", "car")
        else:
            cross_model_name = model_name.replace("car", "ele")

        pre_train_model_path = "./PreTrainModel/" + cross_model_name + "/dev_model"

        if not os.path.exists(pre_train_model_path):
            print("[ERROR] pre-train model isn't exist")
            return

        elem_model = torch.load(pre_train_model_path)

        test_first_process_data_path = "./ModelResult/" + model_name + "/test_first_data_" + str(feature_type) + ".txt"

        if os.path.exists(test_first_process_data_path):
            test_candidate_pair_col, test_pair_representation, test_make_pair_label = \
                shared_utils.read_pickle(test_first_process_data_path)

        else:
            test_candidate_pair_col, test_pair_representation, test_make_pair_label, _, _ = \
                train_test_utils.first_stage_model_test(
                    elem_model, config, test_loader, generate_second_res_eval,
                    eval_parameters=[dataset.test_data_dict['tuple_pair_col']],
                    test_type="gene", feature_type=feature_type
                )

            shared_utils.write_pickle(
                [test_candidate_pair_col, test_pair_representation, test_make_pair_label],
                test_first_process_data_path
            )

        dev_pair_parameters = ["./ModelResult/" + cross_model_name + "/dev_pair_result.txt",
                               "./PreTrainModel/" + cross_model_name + "/dev_pair_model"]

        dev_polarity_parameters = ["./ModelResult/" + cross_model_name + "/dev_polarity_result.txt",
                                   "./PreTrainModel/" + cross_model_name + "/dev_polarity_model"]

        test_pair_parameters = ["./ModelResult/" + cross_model_name + "/test_pair_result.txt", None]
        test_polarity_parameters = ["./ModelResult/" + cross_model_name + "/test_pair_result.txt", None]

        predict_pair_model = torch.load(dev_pair_parameters[1])
        predict_polarity_model = torch.load(dev_polarity_parameters[1])

        test_pair_eval = PairEvaluation(
            config,
            gold_pair_col=dataset.test_data_dict['tuple_pair_col'],
            candidate_pair_col=test_candidate_pair_col,
            elem_col=config.val.elem_col,
            ids_to_tags=config.val.norm_id_map,
            save_model=False
        )

        test_pair_loader = data_loader_utils.get_loader([test_pair_representation], 1)

        train_test_utils.pair_stage_model_test(
            predict_pair_model, config, test_pair_loader, test_pair_eval,
            test_pair_parameters, mode="pair", polarity=False, initialize=(False, False)
        )

        eval_shared_modules.calculate_average_measure(test_pair_eval, global_pair_eval)
        global_pair_eval.avg_model("./ModelResult/" + model_name + "/test_pair_result.txt")
        global_pair_eval.store_result_to_csv([model_name], "result.csv")

        eval_shared_modules.clear_global_measure(global_pair_eval)
        eval_shared_modules.clear_optimize_measure(test_pair_eval)

        # create polarity representation and data loader.
        test_polarity_representation = shared_utils.get_after_pair_representation(test_pair_eval.y_hat, test_pair_representation)
        test_polarity_loader = data_loader_utils.get_loader([test_polarity_representation], 1)

        train_test_utils.pair_stage_model_test(
            predict_polarity_model, config, test_polarity_loader, test_pair_eval,
            test_polarity_parameters, mode="polarity", polarity=True, initialize=(True, True)
        )

        # add average measure.
        eval_shared_modules.calculate_average_measure(test_pair_eval, global_pair_eval)

    elif config.stage_model == "second":
        # 0: 768 + 5, 1: 5, 2: 768
        logger.info("%"*30+"SECOND STAGE RUNNING" + "%"*30)
        feature_type = 0

        # using evaluation to generate index col and pair label.
        generate_second_res_eval = ElementEvaluation(
            config, elem_col=config.val.elem_col,
            ids_to_tags=config.val.invert_norm_id_map
        )

        pre_train_model_path = "./PreTrainModel/" + model_name + "/dev_model"

        if not os.path.exists(pre_train_model_path):
            print("[ERROR] pre-train model isn't exist")
            return

        elem_model = torch.load(pre_train_model_path)

        train_first_process_data_path = "./ModelResult/" + model_name + "/train_first_data_" + str(feature_type) + ".pkl"
        dev_first_process_data_path = "./ModelResult/" + model_name + "/dev_first_data_" + str(feature_type) + ".pkl"
        test_first_process_data_path = "./ModelResult/" + model_name + "/test_first_data_" + str(feature_type) + ".pkl"

        ## extract element for train dataset
        if os.path.exists(train_first_process_data_path):
            train_pair_representation, train_make_pair_label, train_polarity_representation, train_polarity_label = \
                shared_utils.read_pickle(train_first_process_data_path)
        else:
            _, train_pair_representation, train_make_pair_label, train_feature_out, train_bert_feature_out = \
                train_test_utils.first_stage_model_test(
                    elem_model, config, train_loader, generate_second_res_eval,
                    eval_parameters=[dataset.train_data_dict['tuple_pair_col']],
                    test_type="gene", feature_type=feature_type
                )

            train_pair_representation, train_make_pair_label = shared_utils.generate_train_pair_data(
                train_pair_representation, train_make_pair_label
            )

            train_polarity_representation, train_polarity_label = shared_utils.create_polarity_train_data(
                config, dataset.train_data_dict['tuple_pair_col'], train_feature_out,
                train_bert_feature_out, feature_type=feature_type
            )

            shared_utils.write_pickle(
                [train_pair_representation, train_make_pair_label,
                 train_polarity_representation, train_polarity_label],
                train_first_process_data_path
            )

            for i in range(3):
                logger.info("Train pair representation lenght: {}".format(len(train_pair_representation[i])))
                logger.info("Train make pair label: {}".format(train_make_pair_label[i]))
                logger.info("Train polarity label length: {}".format(len(train_polarity_representation[i])))
                logger.info("Train polarity label: ".format(train_polarity_label[i]))

            with open("./ModelResult/" +  "/train_first_data_" + str(feature_type) + ".txt", "w", encoding="utf8") as fout:
                for i, pair in enumerate(train_pair_representation):
                    fout.write(f"{pair}\n{train_make_pair_label[i]}\n{train_polarity_representation[i]}\n{train_polarity_label[i]}\n\n")
        
        ## extract element for dev dataset
        if os.path.exists(dev_first_process_data_path):
            dev_candidate_pair_col, dev_pair_representation, dev_make_pair_label = \
                shared_utils.read_pickle(dev_first_process_data_path)

        else:
            dev_candidate_pair_col, dev_pair_representation, dev_make_pair_label, _, _ = \
                train_test_utils.first_stage_model_test(
                    elem_model, config, dev_loader, generate_second_res_eval,
                    eval_parameters=[dataset.dev_data_dict['tuple_pair_col']],
                    test_type="gene", feature_type=feature_type
                )

            shared_utils.write_pickle(
                [dev_candidate_pair_col, dev_pair_representation, dev_make_pair_label],
                dev_first_process_data_path
            )
            with open("./ModelResult/" +  "/dev_first_data_" + str(feature_type) + ".txt", "w", encoding="utf8") as fout:
                for i, pair in enumerate(dev_candidate_pair_col):
                    fout.write(f"{pair}\n{dev_make_pair_label[i]}\n\n")

        ## extract element for test dataset
        if os.path.exists(test_first_process_data_path):
            test_candidate_pair_col, test_pair_representation, test_make_pair_label = \
                shared_utils.read_pickle(test_first_process_data_path)

        else:
            test_candidate_pair_col, test_pair_representation, test_make_pair_label, _, _ = \
                train_test_utils.first_stage_model_test(
                    elem_model, config, test_loader, generate_second_res_eval,
                    eval_parameters=[dataset.test_data_dict['tuple_pair_col']],
                    test_type="gene", feature_type=feature_type
                )

            shared_utils.write_pickle(
                [test_candidate_pair_col, test_pair_representation, test_make_pair_label],
                test_first_process_data_path
            )
            with open("./ModelResult/" +  "/test_first_data_" + str(feature_type) + ".txt", "w", encoding="utf8") as fout:
                for i, pair in enumerate(test_candidate_pair_col):
                    fout.write(f"{pair}\n{test_make_pair_label[i]}\n\n")

        pair_representation = [train_pair_representation, dev_pair_representation, test_pair_representation]
        make_pair_label = [train_make_pair_label, dev_make_pair_label, test_make_pair_label]

        dev_pair_eval = PairEvaluation(
            config,
            gold_pair_col=dataset.dev_data_dict['tuple_pair_col'],
            candidate_pair_col=dev_candidate_pair_col,
            elem_col=config.val.elem_col,
            ids_to_tags=config.val.norm_id_map,
            save_model=True,
            token_col=[dataset.dev_data_dict['standard_token'], dataset.dev_data_dict['bert_token']]
        )

        test_pair_eval = PairEvaluation(
            config,
            gold_pair_col=dataset.test_data_dict['tuple_pair_col'],
            candidate_pair_col=test_candidate_pair_col,
            elem_col=config.val.elem_col,
            ids_to_tags=config.val.norm_id_map,
            save_model=False,
            token_col=[dataset.test_data_dict['standard_token'], dataset.test_data_dict['bert_token']]
        )

        ## from extracted element tuple, predict comparative label
        train_test_utils.pair_stage_model_main(
            config, dataset, pair_representation, make_pair_label,
            [dev_pair_eval, test_pair_eval, global_pair_eval],
            [train_polarity_representation, train_polarity_label],
            model_parameters, optimizer_parameters, model_name, feature_type
        )


    if config.stage_model == "first":
        global_eval.avg_model("./ModelResult/" + model_name + "/test_extraction_result.txt")
        global_eval.store_result_to_csv([model_name], "result.csv")
    else:
        global_pair_eval.avg_model("./ModelResult/" + model_name + "/test_pair_result.txt")
        global_pair_eval.store_result_to_csv([model_name], "result.csv")


if __name__ == "__main__":
    main()

    


    


    

    
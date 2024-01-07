import torch
import copy
import torch.nn as nn
from data_utils import shared_utils, data_loader_utils
from model_utils import pipeline_model_utils, optimizer_utils
from eval_utils import eval_shared_modules
from tqdm import tqdm

import logging 
logger = logging.getLogger(__name__)


########################################################################################################################
# Train and Test Program
########################################################################################################################
def first_stage_model_train(model, optimizer, train_loader, config, epoch):
    """
    :param model:
    :param optimizer:
    :param train_loader:
    :param config:
    :param epoch:
    :return:
    """
    model.train()
    

    epoch_loss = 0
    for index, data in tqdm(enumerate(train_loader)):
        input_ids, attn_mask, comparative_label, multi_label, result_label = data

        input_ids = torch.tensor(input_ids).long().to(config.device)
        attn_mask = torch.tensor(attn_mask).long().to(config.device)

        comparative_label = torch.tensor(comparative_label).long().to(config.device)
        multi_label = torch.tensor(multi_label).long().to(config.device)
        result_label = torch.tensor(result_label).long().to(config.device)

        # hàm forward (mode train: tính loss, mode eval: extract) trong model baseline
        loss = model(input_ids, attn_mask, comparative_label, multi_label, result_label)

        loss = torch.sum(loss)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logger.info("epoch is {} and Loss: {:.2f}".format(epoch, epoch_loss))


def first_stage_model_test(model, config, test_loader, res_eval, eval_parameters=None, test_type="eval", feature_type=1):
    """
    :param model:
    :param config:
    :param test_loader:
    :param res_eval:
    :param eval_parameters:
    :param test_type: "eval": first stage evaluation, "gene" generate dataset for second and third stage running
    :param feature_type:
    :return:
    """
    elem_feature_embed, result_feature_embed, bert_feature_embed = [], [], []

    assert test_type in {"eval", "gene"}, "[ERROR] test type error!"

    model.eval()
    if test_type == "eval":
        measure_file, model_path = eval_parameters
    else: ## extract task
        gold_pair_label = eval_parameters[0]

    with torch.no_grad():
        for index, data in tqdm(enumerate(test_loader)):
            input_ids, attn_mask, comparative_label, multi_label, result_label = data

            input_ids = torch.tensor(input_ids).long().to(config.device)
            attn_mask = torch.tensor(attn_mask).long().to(config.device)

            ## extract each element of sequence
            bert_feature, elem_feature, elem_output, result_output, sent_output = model(input_ids, attn_mask)

            
            if test_type == "eval":
                ## thêm các giá trị dữ đoán vào result_hat, elem_hat
                res_eval.add_data(elem_output, result_output, attn_mask)
                ## res_eval: self. predict_sent_label
                res_eval.add_sent_label(sent_output)
            else: ## extract 
                res_eval.add_data(elem_output, result_output, attn_mask)
                elem_feature_embed.append(elem_feature)
                bert_feature_embed.append(bert_feature)

    ## đánh giá mô hình hoặc sinh các element
    if test_type == "eval":
        # eval của ElementEvaluation
        res_eval.eval_model(measure_file, model, model_path, multi_elem_score=True)
    else:
        # gold_pair là tuple_pair_col
        return res_eval.generate_elem_representation(
            gold_pair_label, torch.cat(elem_feature_embed, dim=0),
            torch.cat(bert_feature_embed, dim=0), feature_type=feature_type
        )


def pair_stage_model_train(model, optimizer, train_loader, config, epoch):
    """
    :param model:
    :param optimizer:
    :param train_loader:
    :param config:
    :param epoch:
    :return:
    """
    model.train()
    epoch_loss, t = 0, 0
    
    for index, data in tqdm(enumerate(train_loader)):
        pair_representation, pair_label = data

        pair_representation = torch.tensor(pair_representation).float().to(config.device)
        pair_label = torch.tensor(pair_label).long().to(config.device)

        if torch.equal(pair_representation, torch.zeros_like(pair_representation)):
            continue

        loss = model(pair_representation, pair_label)

        loss = torch.sum(loss)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    logger.info("epoch is {} and Loss: {:.2f}".format(epoch, epoch_loss))


def pair_stage_model_test(
        model, config, test_loader, res_eval, eval_parameters=None, mode="pair", polarity=False, initialize=(False, False)):
    """
    :param model: the model
    :param test_loader: test data loader: [input_ids, attn_mask, pos_ids, predicate_label]
    :param config:
    :param res_eval: a Evaluation object
    :param eval_parameters:
    :param mode:
    :param polarity:
    :param initialize:
    :return:
    """
    model.eval()
    measure_file, model_path = eval_parameters

    with torch.no_grad():
        for index, data in tqdm(enumerate(test_loader)):
            pair_representation = data

            pair_representation = torch.tensor(pair_representation).float().to(config.device)

            pair_out = model(pair_representation).view(-1)

            if torch.equal(pair_representation, torch.zeros_like(pair_representation)):
                pair_out = torch.zeros(pair_out.size())

            if mode == "pair":
                res_eval.add_pair_data(pair_out)
            else:
                res_eval.add_polarity_data(pair_out)

    res_eval.eval_model(measure_file, model, model_path, polarity=polarity, initialize=initialize)


########################################################################################################################
# each stage model
########################################################################################################################
def first_stage_model_main(
        config, data_gene, data_loaders, comp_eval, model_parameters, optimizer_parameters, model_name):
    """
    :param config:
    :param data_gene:
    :param data_loaders:
    :param comp_eval:
    :param model_parameters:
    :param optimizer_parameters:
    :param model_name:
    :return:
    """
    train_loader, dev_loader, test_loader = data_loaders
    dev_comp_eval, test_comp_eval, global_comp_eval = comp_eval

    # define first stage model and optimizer
    MODEL2FN = {"bert": pipeline_model_utils.Baseline, "norm": pipeline_model_utils.LSTMModel}

    if "bert" in config.model_mode:
        model = MODEL2FN["bert"](config, model_parameters).to(config.device)
    else:
        # weight = shared_utils.get_pretrain_weight(
        #     config.path.GloVe_path, config.path.Word2Vec_path, data_gene.vocab
        # )
        weight = None
        model = MODEL2FN["norm"](
            config, model_parameters, data_gene.vocab, weight).to(config.device)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        optimizer_need_model = model.module
    else:
        optimizer_need_model = model

    ## 4 Adam optimizer for bert and normal model (LSTM) with different learning rate
    OPTIM2FN = {"bert": optimizer_utils.Baseline_Optim, "norm": optimizer_utils.LSTMModel_Optim}

    if "bert" in config.model_mode:
        optimizer = OPTIM2FN["bert"](optimizer_need_model, optimizer_parameters)
    else:
        optimizer = OPTIM2FN["norm"](optimizer_need_model, optimizer_parameters)

    dev_parameters = ["./ModelResult/" + model_name + "/dev_elem_result.txt",
                      "./PreTrainModel/" + model_name + "/dev_model"]

    # train and test model.
    # using dev để chỉnh tham số?
    logger.info("===============TRAIN FIRST STAGE==================")
    for epoch in range(config.epochs):
        first_stage_model_train(model, optimizer, train_loader, config, epoch)
        first_stage_model_test(model, config, dev_loader, dev_comp_eval, dev_parameters)

    logger.info("=======================TEST FRIST STAGE=========================")
    ## load mô hình trích xuất
    predicate_model = torch.load(dev_parameters[1]) # load pretrain model? có lưu lại model đã train?
    logger.info("Using model {} to predict element.".format(model_name))

    test_parameters = ["./ModelResult/" + model_name + "/test_elem_result.txt", None]

    first_stage_model_test(predicate_model, config, test_loader, test_comp_eval, test_parameters)

    test_comp_eval.print_elem_result(
        data_gene.test_data_dict['input_ids'], data_gene.test_data_dict['attn_mask'],
        "./ModelResult/" + model_name + "/test_result_file" + ".txt", drop_span=False
    )

    # add average measure.
    eval_shared_modules.calculate_average_measure(test_comp_eval, global_comp_eval)


def pair_stage_model_main(config, data_gene, pair_representation, make_pair_label, pair_eval, polarity_col,
                          model_parameters, optimizer_parameters, model_name, feature_type):
    """

    :param config:
    :param pair_representation:
    :param make_pair_label:
    :param pair_eval:
    :param polarity_col:
    :param model_parameters:
    :param optimizer_parameters:
    :param model_name:
    :param feature_type:
    :return:
    """
    train_pair_representation, dev_pair_representation, test_pair_representation = pair_representation
    train_make_pair_label, dev_make_pair_label, test_make_pair_label = make_pair_label
    dev_pair_eval, test_pair_eval, global_pair_eval = pair_eval
    train_polarity_representation, train_polarity_col = polarity_col

    for index in range(10):
        logger.info("Train pair label: {}; Train polarity label: {} ".format(train_make_pair_label[index], train_polarity_col[index]))

    # logger.info("finish second model data generate")

    # get pair loader
    train_pair_loader = data_loader_utils.get_loader([train_pair_representation, train_make_pair_label], 16)
    dev_pair_loader = data_loader_utils.get_loader([dev_pair_representation], 1)
    test_pair_loader = data_loader_utils.get_loader([test_pair_representation], 1)

    # get polarity data loader.
    train_polarity_loader = data_loader_utils.get_loader([train_polarity_representation, train_polarity_col], 16)

    pair_weight = torch.tensor([model_parameters['factor'], 1]).float()

    feature_dim = [4 * (5 + 768), 4 * 5, 4 * 768]
    pair_feature_dim = feature_dim[feature_type]

    # define pair and polarity model.
    # using Logistic Classfier to identify comparative sentence (class = 0, 1)
    pair_model = copy.deepcopy(
        pipeline_model_utils.LogisticClassifier(config, pair_feature_dim, 2, weight=pair_weight).to(config.device)
    )
    # using Logistic Classifier to predict comparative label (9 label in VLSP)
    polarity_model = copy.deepcopy(
        pipeline_model_utils.LogisticClassifier(config, pair_feature_dim, 9).to(config.device)
    )

    if torch.cuda.device_count() > 1:
        pair_model = nn.DataParallel(pair_model)
        polarity_model = nn.DataParallel(polarity_model)
        pair_optimizer = optimizer_utils.Logistic_Optim(pair_model.module, optimizer_parameters)
        polarity_optimizer = optimizer_utils.Logistic_Optim(polarity_model.module, optimizer_parameters)
    else:
        pair_optimizer = optimizer_utils.Logistic_Optim(pair_model, optimizer_parameters)
        polarity_optimizer = optimizer_utils.Logistic_Optim(polarity_model, optimizer_parameters)

    dev_pair_parameters = ["./ModelResult/" + model_name + "/dev_pair_result.txt",
                           "./PreTrainModel/" + model_name + "/dev_pair_model"]

    dev_polarity_parameters = ["./ModelResult/" + model_name + "/dev_polarity_result.txt",
                               "./PreTrainModel/" + model_name + "/dev_polarity_model"]
    logger.info("===============TRAIN SECOND AND THIRD STAGE==================")
    logger.info("===============Identifying comparative sentence==================")
    for epoch in range(config.epochs):
        pair_stage_model_train(pair_model, pair_optimizer, train_pair_loader, config, epoch)
        pair_stage_model_test(
            pair_model, config, dev_pair_loader, dev_pair_eval,
            dev_pair_parameters, mode="pair", polarity=False, initialize=(False, True)
        )

    # get optimize pair model.
    predict_pair_model = torch.load(dev_pair_parameters[1])
    test_pair_parameters = ["./ModelResult/" + model_name + "/test_pair_result.txt", None]
    pair_stage_model_test(
        predict_pair_model, config, dev_pair_loader, dev_pair_eval,
        test_pair_parameters, mode="pair", polarity=False, initialize=(False, False)
    )

    # get representation by is_pair label filter.
    dev_polarity_representation = shared_utils.get_after_pair_representation(dev_pair_eval.y_hat, dev_pair_representation)
    dev_polarity_loader = data_loader_utils.get_loader([dev_polarity_representation], 1)
    logger.info("Dev polarity representation size: {}".format(len(dev_pair_representation[0][0])))
    eval_shared_modules.clear_optimize_measure(dev_pair_eval)

    logger.info("===============Predicting comparative label==================")
    for epoch in range(config.epochs):
        pair_stage_model_train(polarity_model, polarity_optimizer, train_polarity_loader, config, epoch)
        pair_stage_model_test(
            polarity_model, config, dev_polarity_loader, dev_pair_eval,
            dev_polarity_parameters, mode="polarity", polarity=True, initialize=(True, False)
        )

    logger.info("==================TEST SECOND AND THIRD STAGE================")
    predict_pair_model = torch.load(dev_pair_parameters[1])
    predict_polarity_model = torch.load(dev_polarity_parameters[1])

    test_pair_parameters = ["./ModelResult/" + model_name + "/test_pair_result.txt", None]
    test_polarity_parameters = ["./ModelResult/" + model_name + "/test_pair_result.txt", None]

    pair_stage_model_test(
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

    pair_stage_model_test(
        predict_polarity_model, config, test_polarity_loader, test_pair_eval,
        test_polarity_parameters, mode="polarity", polarity=True, initialize=(True, True)
    )

    # test_pair_eval.print_tuple_pair(
    #     data_gene.test_data_dict['standard_token'],
    #     "./ModelResult/" + model_name + "/test_pair_result_file" + ".txt",
    # )

    # predict_tuple_pair_col = test_pair_eval.get_predict_truth_tuple_pair(test_pair_eval.candidate_pair_col)

   
    # add average measure.
    eval_shared_modules.calculate_average_measure(test_pair_eval, global_pair_eval)

    



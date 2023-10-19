import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_utils import layer_utils as Layer


class Baseline(nn.Module):
    def __init__(self, config, model_parameters):
        super(Baseline, self).__init__()
        self.config = config

        self.encoder = Layer.BERTCell(config.model_mode)
        # thay thế embedding bằng các giá trị 0 với tỉ lệ số lượng bằng embed_dropout
        self.embedding_dropout = nn.Dropout(model_parameters['embed_dropout'])

        # define hyper-parameters.
        if "loss_factor" in model_parameters:
            self.gamma = model_parameters['loss_factor']
        else:
            self.gamma = 1

        # define sentence classification
        # out_feature là 2?
        self.sent_linear = nn.Linear(self.encoder.hidden_size, 2)

        # define mapping full-connect layer.
        # changing hidden_representation from hidden_size to (norm_id_map) size (size = 5)
        self.W = nn.ModuleList()
        for i in range(4):
            self.W.append(copy.deepcopy(nn.Linear(self.encoder.hidden_size, len(config.val.norm_id_map))))

        # define multi-crf decode the sequence.
        # mô hình thống kê dự đoán output của sequence, CRF thường được dùng trong POS, NER
        # Trường hợp đầu vào có target, trả về loss CRF để huấn luyện mô hình
        # CRF function receive num_tag = len(norm_id_map) = 5
        self.decoder = nn.ModuleList()
        for i in range(4):
            self.decoder.append(copy.deepcopy(Layer.CRFCell(len(config.val.norm_id_map), batch_first=True)))

    def forward(self, input_ids, attn_mask, comparative_label=None, elem_label=None, result_label=None):
        # get token embedding.
        token_embedding, pooled_output = self.encoder(input_ids, attn_mask)

        batch_size, sequence_length, _ = token_embedding.size()

        final_embedding = self.embedding_dropout(token_embedding)
        class_embedding = self.embedding_dropout(pooled_output)

        # linear mapping.
        # tạo mảng chứa input sequence (last hidden state) ở mô hình học thông tin từ câu đầu vào
        multi_sequence_prob = [self.W[index](final_embedding) for index in range(len(self.W))]
        #class_embedding (batch_size, hidden_size) -> sent_class_prb(batch_size, out_size = 2)
        sent_class_prob = self.sent_linear(class_embedding)

        # decode sequence label, extract các element trong câu
        # decode (CRF) 4 embeddings 
        elem_output = []
        for index in range(3):
            if elem_label is None: # trường hợp eval
                elem_output.append(self.decoder[index](multi_sequence_prob[index], attn_mask, None))
            else:
                elem_output.append(self.decoder[index](multi_sequence_prob[index], attn_mask, elem_label[:, index, :]))

        # result extract sequence label.
        # Kết quả extract opinion của câu
        result_output = self.decoder[3](multi_sequence_prob[3], attn_mask, result_label)

        if elem_label is None and result_label is None: # trường hợp không phải câu comparative
            # trả về max indices sau khi softmax, 0 nếu không phải là comparative seq
            _, sent_output = torch.max(torch.softmax(sent_class_prob, dim=1), dim=1)

            elem_output = torch.cat(elem_output, dim=0).view(3, batch_size, sequence_length).permute(1, 0, 2)

            elem_feature = multi_sequence_prob
            elem_feature = [elem_feature[index].unsqueeze(0) for index in range(len(elem_feature))]
            elem_feature = torch.cat(elem_feature, dim=0).permute(1, 0, 2, 3)
            
            # token_embedding: bert token embedding of each sequence, size (bs, sequence_size, hidden_size)
            # elem_feature: [B, 3, N, feature_dim], 3 list of element extracted per sequence
            # result_feature: [B, N, feature_dim], concat to one list
            # result_output: extraction of result (4) element
            # sent_output: probability of comparative sentence

            return token_embedding, elem_feature, elem_output, result_output, sent_output

        # calculate sent loss and crf loss.
        # cross-entropy loss khi identify comparative seq
        # view(-1), expand tensor to 1 dimension
        sent_loss = F.cross_entropy(sent_class_prob, comparative_label.view(-1))
        # loss khi extract các thực thể = loss crf
        crf_loss = sum(elem_output) + result_output

        # according different model type to get different loss type.
        if self.config.model_type == "classification": # loss khi phân loại câu comparative hoặc non-com
            return sent_loss

        elif self.config.model_type == "extraction": # loss khi extract các thực thể (phục vụ quá trình evaluation)
            return crf_loss

        else: ## loss trong khi training
            return sent_loss + self.gamma * crf_loss


class LSTMModel(nn.Module):
    def __init__(self, config, model_parameters, vocab, weight=None):
        super(LSTMModel, self).__init__()
        self.config = config

        # input embedding.
        if weight is not None:
            self.input_embed = nn.Embedding(len(vocab) + 10, config.input_size).from_pretrained(weight)
        else:
            self.input_embed = nn.Embedding(len(vocab) + 10, config.input_size)

        self.encoder = Layer.LSTMCell(
            config.input_size, config.hidden_size, config.num_layers,
            config.device, batch_first=True, bidirectional=True
        )
        self.embedding_dropout = nn.Dropout(model_parameters['embed_dropout'])

        # define hyper-parameters.
        if "loss_factor" in model_parameters:
            self.gamma = model_parameters['loss_factor']
        else:
            self.gamma = 0

        # define sentence classification
        self.sent_linear = nn.Linear(self.encoder.hidden_size * 2, 2)

        # define mapping full-connect layer.
        self.W = nn.ModuleList()
        for i in range(4):
            self.W.append(copy.deepcopy(nn.Linear(self.encoder.hidden_size * 2, len(config.val.norm_id_map))))

        # define multi-crf decode the sequence.
        self.decoder = nn.ModuleList()
        for i in range(4):
            self.decoder.append(copy.deepcopy(Layer.CRFCell(len(config.val.norm_id_map), batch_first=True)))

    def forward(self, input_ids, attn_mask, comparative_label=None, elem_label=None, result_label=None):
        # get token embedding.

        input_embedding = self.input_embed(input_ids)
        token_embedding, pooled_output = self.encoder(input_embedding)

        batch_size, sequence_length, _ = token_embedding.size()

        final_embedding = self.embedding_dropout(token_embedding)
        class_embedding = self.embedding_dropout(pooled_output)

        # linear mapping.
        # hidden representation of each element 
        multi_sequence_prob = [self.W[index](final_embedding) for index in range(len(self.W))]
        sent_class_prob = self.sent_linear(class_embedding)

        # decode sequence label.
        elem_output = []
        for index in range(3):
            if elem_label is None:
                elem_output.append(self.decoder[index](multi_sequence_prob[index], attn_mask, None))
            else:
                elem_output.append(self.decoder[index](multi_sequence_prob[index], attn_mask, elem_label[:, index, :]))

        # result extract sequence label.
        result_output = self.decoder[3](multi_sequence_prob[3], attn_mask, result_label)

        if elem_label is None and result_label is None:
            ## sent_output: index of Max
            _, sent_output = torch.max(torch.softmax(sent_class_prob, dim=1), dim=1)

            elem_output = torch.cat(elem_output, dim=0).view(3, batch_size, sequence_length).permute(1, 0, 2)

            elem_feature = multi_sequence_prob
            elem_feature = [elem_feature[index].unsqueeze(0) for index in range(len(elem_feature))]
            elem_feature = torch.cat(elem_feature, dim=0).permute(1, 0, 2, 3)

            # elem_feature: [B, 3, N, feature_dim]
            # result_feature: [B, N, feature_dim]
            return token_embedding, elem_feature, elem_output, result_output, sent_output

        # calculate sent loss and crf loss.
        sent_loss = F.cross_entropy(sent_class_prob, comparative_label.view(-1))
        crf_loss = sum(elem_output) + result_output

        print(sent_loss, crf_loss)
        # according different model type to get different loss type.
        if self.config.model_type == "classification":
            return sent_loss

        elif self.config.model_type == "extraction":
            return crf_loss

        else:
            return sent_loss + self.gamma * crf_loss


class LogisticClassifier(nn.Module):
    def __init__(self, config, feature_dim, class_num=2, dropout=0.1, weight=None):
        super(LogisticClassifier, self).__init__()
        self.config = config
        self.class_num = class_num

        self.feature_dim = feature_dim
        self.fc = nn.Linear(feature_dim, class_num)
        self.weight = weight

        self.dropout = nn.Dropout(dropout)

    def forward(self, pair_representation, pair_label=None):
        predict_label = self.fc(pair_representation.view(-1, self.feature_dim))
        predict_label = self.dropout(predict_label)
        print("Predict {} - Truth {}".format(predict_label, pair_label))

        # weight = torch.tensor([1, 1, 1, 1]).float().to(self.config.device)
        # calculate loss.
        if pair_label is not None:
            if self.weight is not None:
                self.weight = self.weight.to(self.config.device)
                criterion = nn.CrossEntropyLoss(weight=self.weight)
            else:
                criterion = nn.CrossEntropyLoss()
            
            return criterion(predict_label, pair_label.view(-1))
        else:
            return torch.max(F.softmax(predict_label, dim=-1), dim=-1)[-1]


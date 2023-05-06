import logging
from modules import *
import torch.nn.functional as F


class EntityEncoder(nn.Module):
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None, dropout_input=0.3, finetune=False,
                 dropout_neighbors=0.0,
                 device=torch.device("cpu")):
        super(EntityEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.pad_idx = num_symbols
        self.symbol_emb = nn.Embedding(num_symbols + 1, embed_dim, padding_idx=self.pad_idx).cuda()
        self.num_symbols = num_symbols

        self.gcn_w = nn.Linear(2 * self.embed_dim, self.embed_dim).cuda()
        self.gcn_b = nn.Parameter(torch.FloatTensor(self.embed_dim)).cuda()
        self.dropout = nn.Dropout(dropout_input)
        init.xavier_normal_(self.gcn_w.weight)
        init.constant_(self.gcn_b, 0)
        self.pad_tensor = torch.tensor([self.pad_idx], requires_grad=False).cuda()

        if use_pretrain:
            logging.info('LOADING KB EMBEDDINGS')
            self.symbol_emb.weight.data.copy_(torch.from_numpy(embed))
            if not finetune:
                logging.info('FIX KB EMBEDDING')
                self.symbol_emb.weight.requires_grad = False

        self.NeighborAggregator = AttentionSelectContext(dim=embed_dim, dropout=dropout_neighbors)

    def neighbor_encoder_mean(self, connections, num_neighbors):
        """
        connections: (batch, 200, 2)
        num_neighbors: (batch,)
        """
        num_neighbors = num_neighbors.unsqueeze(1)
        relations = connections[:, :, 0].squeeze(-1)
        entities = connections[:, :, 1].squeeze(-1)
        rel_embeds = self.dropout(self.symbol_emb(relations))
        ent_embeds = self.dropout(self.symbol_emb(entities))

        concat_embeds = torch.cat((rel_embeds, ent_embeds), dim=-1)
        out = self.gcn_w(concat_embeds)

        out = torch.sum(out, dim=1)
        out = out / num_neighbors
        return out.tanh()

    def neighbor_encoder_soft_select(self, connections_left, connections_right, head_left, head_right):
        """
        :param connections_left: [b, max, 2]
        :param connections_right:
        :param head_left:
        :param head_right:
        :return:
        """
        # relations_left是left的邻居关系，entities_left是left的邻居实体 [5,50,2] -> 两个[5,50]
        relations_left = connections_left[:, :, 0].squeeze(-1)
        entities_left = connections_left[:, :, 1].squeeze(-1)
        # 将关系和实体换成嵌入表示 [5, 50] -> [5, 50,100]
        # todo 不进行dropout试试
        rel_embeds_left = self.symbol_emb(relations_left) # [b, max, dim]
        ent_embeds_left = self.symbol_emb(entities_left)
        # rel_embeds_left = self.dropout(self.symbol_emb(relations_left))  # [b, max, dim]
        # ent_embeds_left = self.dropout(self.symbol_emb(entities_left))

        # tensor.expand_as()这个函数就是把一个tensor变成和函数括号内一样形状的tensor,此处得到一个数值全部为pad_id
        pad_matrix_left = self.pad_tensor.expand_as(relations_left)
        #
        mask_matrix_left = torch.eq(relations_left, pad_matrix_left).squeeze(-1)  # [b, max]

        relations_right = connections_right[:, :, 0].squeeze(-1)
        entities_right = connections_right[:, :, 1].squeeze(-1)
        # todo 不进行dropout试试
        rel_embeds_right = self.symbol_emb(relations_right)  # (batch, 200, embed_dim)
        ent_embeds_right = self.symbol_emb(entities_right)
        # rel_embeds_right = self.dropout(self.symbol_emb(relations_right))  # (batch, 200, embed_dim)
        # ent_embeds_right = self.dropout(self.symbol_emb(entities_right))  # (batch, 200, embed_dim)

        pad_matrix_right = self.pad_tensor.expand_as(relations_right)
        mask_matrix_right = torch.eq(relations_right, pad_matrix_right).squeeze(-1)  # [b, max]
        # head_left是实体对中左边实体的嵌入，rel_embeds_left是实体对中左边实体的邻居关系的嵌入
        # ent_embeds_left是实体对中左边实体的邻居实体的嵌入
        left = [head_left, rel_embeds_left, ent_embeds_left]
        right = [head_right, rel_embeds_right, ent_embeds_right]
        # 注意力机制：邻居聚合器
        output = self.NeighborAggregator(left, right, mask_matrix_left, mask_matrix_right)
        return output

    def forward(self, entity, entity_meta=None):
        '''
         query: (batch_size, 2)
         entity: (few, 2)
         return: (batch_size, )
         '''
        if entity_meta is not None:
            # 将entity换成嵌入表示
            entity = self.symbol_emb(entity)
            entity_left_connections, entity_left_degrees, entity_right_connections, entity_right_degrees = entity_meta

            # entity_left、entity_right也是嵌入表示
            entity_left, entity_right = torch.split(entity, 1, dim=1)
            entity_left = entity_left.squeeze(1)
            entity_right = entity_right.squeeze(1)

            # self.neighbor_encoder_soft_select返回的是 left, right
            entity_left, entity_right = self.neighbor_encoder_soft_select(entity_left_connections,
                                                                          entity_right_connections,
                                                                          entity_left, entity_right)
        else:
            # no_meta
            entity = self.symbol_emb(entity)
            entity_left, entity_right = torch.split(entity, 1, dim=1)
            entity_left = entity_left.squeeze(1)
            entity_right = entity_right.squeeze(1)
        return entity_left, entity_right


class RelationRepresentation(nn.Module):
    def __init__(self, emb_dim, num_transformer_layers, num_transformer_heads, dropout_rate=0.1):
        super(RelationRepresentation, self).__init__()
        self.RelationEncoder = TransformerEncoder(model_dim=emb_dim, ffn_dim=emb_dim * num_transformer_heads * 2,
                                                  num_heads=num_transformer_heads, dropout=dropout_rate,
                                                  num_layers=num_transformer_layers, max_seq_len=4,
                                                  with_pos=True)

    def forward(self, left, right):
    # def forward(self, left, right, tem):
        """
        forward
        :param left: [batch, dim]
        :param right: [batch, dim]

        :return: [batch, dim]
        """

        relation = self.RelationEncoder(left, right)
        # relation = self.RelationEncoder(left, right, tem)
        return relation


# todo 时间编码器
class TimeEncoder(nn.Module):
    def __init__(self, embed_dim, tem_total):
        super(TimeEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.tem_total = tem_total

        self.time_emb = nn.Embedding(tem_total, embed_dim)
        tem_weight = torch.FloatTensor(self.tem_total, self.embed_dim)
        nn.init.xavier_uniform(tem_weight)
        self.time_emb.weight = nn.Parameter(tem_weight)
        normalize_temporal_emb = F.normalize(self.time_emb.weight.data, p=2, dim=1)
        self.time_emb.weight.data = normalize_temporal_emb

    def forward(self, time):
        time = self.time_emb(time)
        return time


class Matcher(nn.Module):
    def __init__(self, embed_dim, num_symbols, use_pretrain=True, embed=None, dropout_layers=0.1, dropout_input=0.3,
                 dropout_neighbors=0.0,
                 finetune=False, num_transformer_layers=6, num_transformer_heads=4,
                 device=torch.device("cpu")
                 ):
        super(Matcher, self).__init__()
        self.EntityEncoder = EntityEncoder(embed_dim, num_symbols,
                                           use_pretrain=use_pretrain,
                                           embed=embed, dropout_input=dropout_input,
                                           dropout_neighbors=dropout_neighbors,
                                           finetune=finetune, device=device)
        self.RelationRepresentation = RelationRepresentation(emb_dim=embed_dim,
                                                             num_transformer_layers=num_transformer_layers,
                                                             num_transformer_heads=num_transformer_heads,
                                                             dropout_rate=dropout_layers)
        self.Prototype = SoftSelectPrototype(embed_dim * num_transformer_heads)

        # todo 添加时间编码器
        self.TimeEncoder = TimeEncoder(embed_dim, 304)

    def forward(self, support, query, false=None, isEval=False, support_meta=None, query_meta=None, false_meta=None):
    # def forward(self, support, query, false=None, isEval=False, support_meta=None, query_meta=None, false_meta=None, support_t=None, query_t=None, false_t=None):
        """
        :param support:
        :param query:
        :param false:
        :param isEval:
        :param support_meta:
        :param query_meta:
        :param false_meta:
        :return:
        """

        # support_t = None
        # query_t = None
        # false_t = None

        # todo 以一种方式加入时间的嵌入表示

        if not isEval:
            # 论文中的 Adaptive Neighbor Encoder for Entities
            # EntityEncoder返回的是 entity_left, entity_right
            support_r = self.EntityEncoder(support, support_meta)
            query_r = self.EntityEncoder(query, query_meta)
            false_r = self.EntityEncoder(false, false_meta)


            # todo 自己添加的用来对时间进行编码的模块
            # support_t = self.TimeEncoder(support_t)
            # query_t = self.TimeEncoder(query_t)
            # false_t = self.TimeEncoder(false_t)

            # 论文中的 Transformer Encoder for Entity Pairs
            support_r = self.RelationRepresentation(support_r[0], support_r[1])
            query_r = self.RelationRepresentation(query_r[0], query_r[1])
            false_r = self.RelationRepresentation(false_r[0], false_r[1])

            # todo Transformer Encoder中加入时间
            # support_r = self.RelationRepresentation(support_r[0], support_r[1], support_t)
            # query_r = self.RelationRepresentation(query_r[0], query_r[1], query_t)
            # false_r = self.RelationRepresentation(false_r[0], false_r[1], false_t)

            # 论文中的 Adaptive Matching Processor
            center_q = self.Prototype(support_r, query_r)
            center_f = self.Prototype(support_r, false_r)

            # 公式（11）
            positive_score = torch.sum(query_r * center_q, dim=1)
            negative_score = torch.sum(false_r * center_f, dim=1)
        else:

            support_r = self.EntityEncoder(support, support_meta)
            query_r = self.EntityEncoder(query, query_meta)

            # todo 自己添加的用来对时间进行编码的模块
            # support_t = self.TimeEncoder(support_t)
            # query_t = self.TimeEncoder(query_t)

            support_r = self.RelationRepresentation(support_r[0], support_r[1])
            query_r = self.RelationRepresentation(query_r[0], query_r[1])
            # #
            # support_r = self.RelationRepresentation(support_r[0], support_r[1], support_t)
            # query_r = self.RelationRepresentation(query_r[0], query_r[1], query_t)

            center_q = self.Prototype(support_r, query_r)
            positive_score = torch.sum(query_r * center_q, dim=1)
            negative_score = None
        return positive_score, negative_score

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

from collections import defaultdict
from torch import optim
from collections import deque
from args import read_options
from data_loader import *
from matcher import *
from tensorboardX import SummaryWriter
from tqdm import tqdm
import random

class Trainer(object):
    def __init__(self, arg):
        super(Trainer, self).__init__()
        for k, v in vars(arg).items():
            setattr(self, k, v)

        #    todo 换成CPU运行
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # self.device = torch.device("cpu")

        self.meta = not self.no_meta

        # pre-train
        if self.random_embed:
            use_pretrain = False
        else:
            use_pretrain = True

        logging.info('LOADING SYMBOL ID AND SYMBOL EMBEDDING')
        if self.test or self.random_embed:
            # gen symbol2id, without embedding
            self.load_symbol2id()
            use_pretrain = False
        else:
            self.load_embed()
        self.use_pretrain = use_pretrain

        self.num_symbols = len(self.symbol2id.keys()) - 1  # one for 'PAD'
        # id是从0开始的，比num_symbols少一个
        self.pad_id = self.num_symbols

        self.Matcher = Matcher(self.embed_dim, self.num_symbols,
                               use_pretrain=self.use_pretrain,
                               embed=self.symbol2vec,
                               dropout_layers=self.dropout_layers,
                               dropout_input=self.dropout_input,
                               dropout_neighbors=self.dropout_neighbors,
                               finetune=self.fine_tune,
                               num_transformer_layers=self.num_transformer_layers,
                               num_transformer_heads=self.num_transformer_heads,
                               device=self.device
                               )

        self.Matcher.cuda()
        self.batch_nums = 0
        if self.test:
            self.writer = None
        else:
            self.writer = SummaryWriter('logs/' + self.prefix)

        self.parameters = filter(lambda p: p.requires_grad, self.Matcher.parameters())

        self.optim = optim.Adam(self.parameters, lr=self.lr, weight_decay=self.weight_decay)
        self.ent2id = json.load(open(self.dataset + '/ent2ids_shuffle'))
        self.num_ents = len(self.ent2id.keys())

        logging.info('BUILDING CONNECTION MATRIX')
        degrees = self.build_connection(max_=self.max_neighbor)

        logging.info('LOADING CANDIDATES ENTITIES')
        self.rel2candidates = json.load(open(self.dataset + '/rel2candidates_all_ent.json'))

        # load answer dict
        self.e1rel_e2 = defaultdict(list)
        self.e1rel_e2 = json.load(open(self.dataset + '/e1rel_e2.json'))

    def load_symbol2id(self):
        # gen symbol2id, without embedding
        symbol_id = {}
        rel2id = json.load(open(self.dataset + '/rel2ids_shuffle'))
        ent2id = json.load(open(self.dataset + '/ent2ids_shuffle'))
        i = 0
        # rel and ent combine together
        for key in rel2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1

        for key in ent2id.keys():
            if key not in ['', 'OOV']:
                symbol_id[key] = i
                i += 1

        symbol_id['PAD'] = i
        self.symbol2id = symbol_id
        self.symbol2vec = None

    def load_embed(self):
        # gen symbol2id, with embedding
        symbol_id = {}
        rel2id = json.load(open(self.dataset + '/rel2ids_shuffle'))  # relation2id contains inverse rel
        ent2id = json.load(open(self.dataset + '/ent2ids_shuffle'))

        logging.info('LOADING PRE-TRAINED EMBEDDING')
        if self.embed_model in ['DistMult', 'TransE', 'ComplEx', 'RESCAL']:
            ent_embed = np.loadtxt(self.dataset + '/entity2vec.' + self.embed_model)
            rel_embed = np.loadtxt(self.dataset + '/relation2vec.' + self.embed_model)  # contain inverse edge

            if self.embed_model == 'ComplEx':
                # normalize the complex embeddings
                ent_mean = np.mean(ent_embed, axis=1, keepdims=True)
                ent_std = np.std(ent_embed, axis=1, keepdims=True)
                rel_mean = np.mean(rel_embed, axis=1, keepdims=True)
                rel_std = np.std(rel_embed, axis=1, keepdims=True)
                eps = 1e-3
                ent_embed = (ent_embed - ent_mean) / (ent_std + eps)
                rel_embed = (rel_embed - rel_mean) / (rel_std + eps)

            assert ent_embed.shape[0] == len(ent2id.keys())
            assert rel_embed.shape[0] == len(rel2id.keys())

            i = 0
            embeddings = []
            for key in rel2id.keys():
                if key not in ['', 'OOV']:
                    symbol_id[key] = i
                    i += 1
                    embeddings.append(list(rel_embed[rel2id[key], :]))

            for key in ent2id.keys():
                if key not in ['', 'OOV']:
                    symbol_id[key] = i
                    i += 1
                    embeddings.append(list(ent_embed[ent2id[key], :]))

            symbol_id['PAD'] = i
            embeddings.append(list(np.zeros((rel_embed.shape[1],))))
            embeddings = np.array(embeddings)
            assert embeddings.shape[0] == len(symbol_id.keys())

            self.symbol2id = symbol_id
            self.symbol2vec = embeddings

    def build_connection(self, max_=100):
        self.connections = (np.ones((self.num_ents, max_, 2)) * self.pad_id).astype(int)
        self.e1_rele2 = defaultdict(list)
        self.e1_degrees = defaultdict(int)
        with open(self.dataset + '/path_graph', encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines):
                e1, rel, e2, t = line.rstrip().split('\t')
                self.e1_rele2[e1].append((self.symbol2id[rel], self.symbol2id[e2]))  # 1-n
                self.e1_rele2[e2].append((self.symbol2id[rel + '_inv'], self.symbol2id[e1]))  # n-1

        degrees = {}
        for ent, id_ in self.ent2id.items():
            neighbors = self.e1_rele2[ent]
            if len(neighbors) > max_:
                neighbors = neighbors[:max_]
            degrees[ent] = len(neighbors)
            self.e1_degrees[id_] = len(neighbors)  # add one for self conn
            for idx, _ in enumerate(neighbors):
                self.connections[id_, idx, 0] = _[0]  # rel
                self.connections[id_, idx, 1] = _[1]  # tail
        return degrees

    def save(self, path=None):
        if not path:
            path = self.save_path
        torch.save(self.Matcher.state_dict(), path)

    def load(self, path=None):
        if path:
            self.Matcher.load_state_dict(torch.load(path))
        else:
            self.Matcher.load_state_dict(torch.load(self.save_path))

    def get_meta(self, left, right):
        left_connections = Variable(
            torch.LongTensor(np.stack([self.connections[_, :, :] for _ in left], axis=0))).cuda()
        left_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in left])).cuda()
        right_connections = Variable(
            torch.LongTensor(np.stack([self.connections[_, :, :] for _ in right], axis=0))).cuda()
        right_degrees = Variable(torch.FloatTensor([self.e1_degrees[_] for _ in right])).cuda()
        return (left_connections, left_degrees, right_connections, right_degrees)

    def train(self):
        logging.info('START TRAINING...')
        best_mrr = 0.0
        best_batches = 0

        losses = deque([], self.log_every)
        margins = deque([], self.log_every)
        for data in train_generate(self.dataset, self.batch_size, self.train_few, self.symbol2id, self.ent2id,
                                   self.e1rel_e2):
            support, query, false, support_left, support_right, query_left, query_right, false_left, false_right = data

            self.batch_nums += 1
            support_meta = self.get_meta(support_left, support_right)
            query_meta = self.get_meta(query_left, query_right)
            false_meta = self.get_meta(false_left, false_right)

            support = Variable(torch.LongTensor(support)).cuda()
            query = Variable(torch.LongTensor(query)).cuda()
            false = Variable(torch.LongTensor(false)).cuda()

            self.Matcher.train()
            if self.no_meta:
                positive_score, negative_score = self.Matcher(support, query, false, isEval=False)
            else:
                positive_score, negative_score = self.Matcher(support, query, false, isEval=False,
                                                              support_meta=support_meta,
                                                              query_meta=query_meta,
                                                              false_meta=false_meta)
            margin_ = positive_score - negative_score
            loss = F.relu(self.margin - margin_).mean()
            margins.append(margin_.mean().item())
            lr = adjust_learning_rate(optimizer=self.optim, epoch=self.batch_nums, lr=self.lr,
                                      warm_up_step=self.warm_up_step,
                                      max_update_step=self.max_batches)
            losses.append(loss.item())

            # Adam优化器
            self.optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm(self.parameters, self.grad_clip)
            self.optim.step()
            if self.batch_nums % self.log_every == 0:
                lr = self.optim.param_groups[0]['lr']
                logging.info(
                    'Batch: {:d}, Avg_batch_loss: {:.6f}, lr: {:.6f}, '.format(
                        self.batch_nums,
                        np.mean(losses),
                        lr))
                self.writer.add_scalar('Avg_batch_loss_every_log', np.mean(losses), self.batch_nums)

            if self.batch_nums % self.eval_every == 0:
                logging.info('Batch_nums is %d' % self.batch_nums)
                hits10, hits5, hits1, mrr = self.eval(meta=self.meta, mode='dev')
                self.writer.add_scalar('HITS10', hits10, self.batch_nums)
                self.writer.add_scalar('HITS5', hits5, self.batch_nums)
                self.writer.add_scalar('HITS1', hits1, self.batch_nums)
                self.writer.add_scalar('MRR', mrr, self.batch_nums)
                self.save()

                if mrr > best_mrr:
                    self.save(self.save_path + '_best')
                    best_mrr = mrr
                    best_batches = self.batch_nums
                logging.info('Best_mrr is {:.6f}, when batch num is {:d}'.format(best_mrr, best_batches))

            if self.batch_nums == self.max_batches:
                self.save()
                break

            if self.batch_nums - best_batches > self.eval_every * 10:
                logging.info('Early stop!')
                self.save()
                break

    def initialize_model_directory(args, random_seed=None):
        # add model parameter info to model directory
        model_root_dir = args.model_root_dir
        dataset = os.path.basename(os.path.normpath(args.data_dir))

        reverse_edge_tag = '-RV' if args.add_reversed_training_edges else ''
        entire_graph_tag = '-EG' if args.train_entire_graph else ''
        if args.xavier_initialization:
            initialization_tag = '-xavier'
        elif args.uniform_entity_initialization:
            initialization_tag = '-uniform'
        else:
            initialization_tag = ''

        # Hyperparameter signature
        if args.model in ['rule']:
            hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
                args.baseline,
                args.entity_dim,
                args.relation_dim,
                args.history_num_layers,
                args.learning_rate,
                args.emb_dropout_rate,
                args.ff_dropout_rate,
                args.action_dropout_rate,
                args.bandwidth,
                args.beta
            )
        elif args.model.startswith('point'):
            if args.baseline == 'avg_reward':
                print('* Policy Gradient Baseline: average reward')
            elif args.baseline == 'avg_reward_normalized':
                print('* Policy Gradient Baseline: average reward baseline plus normalization')
            else:
                print('* Policy Gradient Baseline: None')
            if args.action_dropout_anneal_interval < 1000:
                hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
                    args.baseline,
                    args.entity_dim,
                    args.relation_dim,
                    args.history_num_layers,
                    args.learning_rate,
                    args.emb_dropout_rate,
                    args.ff_dropout_rate,
                    args.action_dropout_rate,
                    args.action_dropout_anneal_factor,
                    args.action_dropout_anneal_interval,
                    args.bandwidth,
                    args.beta
                )
                if args.mu != 1.0:
                    hyperparam_sig += '-{}'.format(args.mu)
            else:
                hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
                    args.baseline,
                    args.entity_dim,
                    args.relation_dim,
                    args.history_num_layers,
                    args.learning_rate,
                    args.emb_dropout_rate,
                    args.ff_dropout_rate,
                    args.action_dropout_rate,
                    args.bandwidth,
                    args.beta
                )
            if args.reward_shaping_threshold > 0:
                hyperparam_sig += '-{}'.format(args.reward_shaping_threshold)
        elif args.model == 'distmult' or args.model == 'TransE':
            hyperparam_sig = '{}-{}-{}-{}-{}'.format(
                args.entity_dim,
                args.relation_dim,
                args.learning_rate,
                args.emb_dropout_rate,
                args.label_smoothing_epsilon
            )
        elif args.model == 'complex':
            hyperparam_sig = '{}-{}-{}-{}-{}'.format(
                args.entity_dim,
                args.relation_dim,
                args.learning_rate,
                args.emb_dropout_rate,
                args.label_smoothing_epsilon
            )
        elif args.model in ['conve', 'hypere', 'triplee']:
            hyperparam_sig = '{}-{}-{}-{}-{}-{}-{}-{}-{}'.format(
                args.entity_dim,
                args.relation_dim,
                args.learning_rate,
                args.num_out_channels,
                args.kernel_size,
                args.emb_dropout_rate,
                args.hidden_dropout_rate,
                args.feat_dropout_rate,
                args.label_smoothing_epsilon
            )
        else:
            raise NotImplementedError

        model_sub_dir = '{}-{}{}{}{}-{}'.format(
            dataset,
            args.model,
            reverse_edge_tag,
            entire_graph_tag,
            initialization_tag,
            hyperparam_sig
        )
        if args.model == 'set':
            model_sub_dir += '-{}'.format(args.beam_size)
            model_sub_dir += '-{}'.format(args.num_paths_per_entity)
        if args.relation_only:
            model_sub_dir += '-ro'
        elif args.relation_only_in_path:
            model_sub_dir += '-rpo'
        elif args.type_only:
            model_sub_dir += '-to'

        if args.test:
            model_sub_dir += '-test'

        if random_seed:
            model_sub_dir += '.{}'.format(random_seed)

        model_dir = os.path.join(model_root_dir, model_sub_dir)

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
            print('Model directory created: {}'.format(model_dir))
        else:
            print('Model directory exists: {}'.format(model_dir))

        args.model_dir = model_dir

    def construct_model(args):
        """
        Construct NN graph.
        """
        kg = KnowledgeGraph(args)
        if args.model.endswith('.gc'):
            kg.load_fuzzy_facts()

        if args.model in ['point', 'point.gc']:
            pn = GraphSearchPolicy(args)
            lf = PolicyGradient(args, kg, pn)
        elif args.model.startswith('point.rs'):
            pn = GraphSearchPolicy(args)
            fn_model = args.model.split('.')[-1]
            fn_args = copy.deepcopy(args)
            fn_args.model = fn_model
            fn_args.relation_only = False
            if fn_model == 'complex':
                fn = ComplEx(fn_args)
                fn_kg = KnowledgeGraph(fn_args)
            elif fn_model == 'distmult':
                fn = DistMult(fn_args)
                fn_kg = KnowledgeGraph(fn_args)
            elif fn_model == 'conve':
                fn = ConvE(fn_args, kg.num_entities)
                fn_kg = KnowledgeGraph(fn_args)
            lf = RewardShapingPolicyGradient(args, kg, pn, fn_kg, fn)
        elif args.model == 'complex':
            fn = ComplEx(args)
            lf = EmbeddingBasedMethod(args, kg, fn)
        elif args.model == 'distmult':
            fn = DistMult(args)
            lf = EmbeddingBasedMethod(args, kg, fn)
        elif args.model == 'conve':
            fn = ConvE(args, kg.num_entities)
            lf = EmbeddingBasedMethod(args, kg, fn)
        elif args.model == 'TransE':
            fn = TransE(args)
            lf = EmbeddingBasedMethod(args, kg, fn)
        else:
            raise NotImplementedError
        return lf

    def eval(self, mode='dev', meta=False):
        self.Matcher.eval()

        symbol2id = self.symbol2id
        few = self.few

        logging.info('EVALUATING ON %s DATA' % mode.upper())
        if mode == 'dev':
            test_tasks = json.load(open(self.dataset + '/dev_tasks.json'))
        else:
            test_tasks = json.load(open(self.dataset + '/test_tasks.json'))

        rel2candidates = self.rel2candidates
        # todo
        time2id = json.load(open('./icews18' + '/time2id.json'))

        hits10 = []
        hits5 = []
        hits1 = []
        mrr = []
        for query_ in test_tasks.keys():
            hits10_ = []
            hits5_ = []
            hits1_ = []
            mrr_ = []

            candidates = rel2candidates['0']
            candidates = random.sample(candidates, 20000)

            # candidates = rel2candidates[query_]
            support_triples = test_tasks[query_][:few]
            support_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in support_triples]

            if meta:
                support_left = [self.ent2id[triple[0]] for triple in support_triples]
                support_right = [self.ent2id[triple[2]] for triple in support_triples]
                support_meta = self.get_meta(support_left, support_right)

                # todo 2021年6月23日11:23:13
                s_t = [time2id[str(triple[3])] for triple in support_triples]

            support = Variable(torch.LongTensor(support_pairs)).cuda()



            for triple in test_tasks[query_][few:]:
                true = triple[2]
                query_pairs = []
                query_pairs.append([symbol2id[triple[0]], symbol2id[triple[2]]])

                # todo
                t = time2id[str(triple[3])]

                if meta:
                    query_left = []
                    query_right = []
                    query_t = []
                    query_left.append(self.ent2id[triple[0]])
                    query_right.append(self.ent2id[triple[2]])


                for ent in candidates:
                    if (ent not in self.e1rel_e2[triple[0] + triple[1]]) and ent != true:
                        query_pairs.append([symbol2id[triple[0]], symbol2id[ent]])
                        if meta:
                            query_left.append(self.ent2id[triple[0]])
                            query_right.append(self.ent2id[ent])

                query = Variable(torch.LongTensor(query_pairs)).cuda()

                # todo
                support_t = Variable(torch.LongTensor(s_t)).cuda()
                query_t.append(t)
                query_t = Variable(torch.LongTensor(query_t)).cuda()

                if meta:
                    query_meta = self.get_meta(query_left, query_right)
                    scores, _ = self.Matcher(support, query, None, isEval=True,
                                             support_meta=support_meta,
                                             query_meta=query_meta,
                                             false_meta=None)

                    scores.detach()
                    scores = scores.data

                scores = scores.cpu().numpy()
                sort = list(np.argsort(scores, kind='stable'))[::-1]
                rank = sort.index(0) + 1
                if rank <= 10:
                # if rank <= 1000:
                    hits10.append(1.0)
                    hits10_.append(1.0)
                else:
                    hits10.append(0.0)
                    hits10_.append(0.0)
                if rank <= 5:
                    hits5.append(1.0)
                    hits5_.append(1.0)
                else:
                    hits5.append(0.0)
                    hits5_.append(0.0)
                if rank <= 1:
                    hits1.append(1.0)
                    hits1_.append(1.0)
                else:
                    hits1.append(0.0)
                    hits1_.append(0.0)
                mrr.append(1.0 / rank)
                mrr_.append(1.0 / rank)
        logging.critical('HITS10: {:.3f}'.format(np.mean(hits10)))
        logging.critical('HITS5: {:.3f}'.format(np.mean(hits5)))
        logging.critical('HITS1: {:.3f}'.format(np.mean(hits1)))
        logging.critical('MRR: {:.3f}'.format(np.mean(mrr)))
        return np.mean(hits10), np.mean(hits5), np.mean(hits1), np.mean(mrr)

    def test_(self, path=None):
        self.load(path)
        logging.info('Pre-trained model loaded for test')
        self.eval(mode='test', meta=self.meta)

    def eval_(self, path=None):
        self.load(path)
        logging.info('Pre-trained model loaded for dev')
        self.eval(mode='dev', meta=self.meta)


def seed_everything(seed=2040):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def adjust_learning_rate(optimizer, epoch, lr, warm_up_step, max_update_step, end_learning_rate=0.0, power=1.0):
    epoch += 1
    if warm_up_step > 0 and epoch <= warm_up_step:
        warm_up_factor = epoch / float(warm_up_step)
        lr = warm_up_factor * lr
    elif epoch >= max_update_step:
        lr = end_learning_rate
    else:
        lr_range = lr - end_learning_rate
        pct_remaining = 1 - (epoch - warm_up_step) / (max_update_step - warm_up_step)
        lr = lr_range * (pct_remaining ** power) + end_learning_rate

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


if __name__ == '__main__':

    args = read_options()
    if not os.path.exists('./logs_'):
        os.mkdir('./logs_')
    if not os.path.exists('./logs'):
        os.mkdir('./logs')

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler('./logs_/log-{}.txt'.format(args.prefix))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    seed_everything(args.seed)

    logging.info('*' * 100)
    logging.info('*** hyper-parameters ***')
    for k, v in vars(args).items():
        logging.info(k + ': ' + str(v))
    logging.info('*' * 100)

    trainer = Trainer(args)

    if args.test:
        trainer.test_()
        trainer.eval_()
    else:
        trainer.train()
        print('last checkpoint!')
        trainer.eval_()
        trainer.test_()
        print('best checkpoint!')
        trainer.eval_(args.save_path + '_best')
        trainer.test_(args.save_path + '_best')

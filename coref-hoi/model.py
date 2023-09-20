import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AutoTokenizer
import util
import logging
from collections import Iterable
import numpy as np
import torch.nn.init as init
import higher_order as ho
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import fasttext
import fasttext.util




logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger()


class CorefModel(nn.Module):
    def __init__(self, config, device, num_genres=None):
        super().__init__()
        self.config = config
        self.device = device

        self.num_genres = num_genres if num_genres else len(config['genres'])
        self.max_seg_len = config['max_segment_len']
        self.max_span_width = config['max_span_width']
        assert config['loss_type'] in ['marginalized', 'hinge']
        if config['coref_depth'] > 1 or config['higher_order'] == 'cluster_merging':
            assert config['fine_grained']  # Higher-order is in slow fine-grained scoring

        # Model
        self.dropout = nn.Dropout(p=config['dropout_rate'])
        self.bert = BertModel.from_pretrained(config['bert_pretrained_name_or_path'])
        self.tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

        self.bert_emb_size = self.bert.config.hidden_size

        if config['add_numberbatch_emb']:
            self.numberbatch_dict = self.embeddings_index(config['numberbatch_path'])
            self.bert_emb_size = self.bert_emb_size + 300
        if config['add_glove_emb']:
            self.glove_dict = self.embeddings_index(config['glove_path'])
            self.bert_emb_size = self.bert_emb_size + 300
        if config['add_fasttext_emb']:
            fasttext.util.download_model('en', if_exists='ignore')
            self.fasttext = fasttext.load_model('cc.en.300.bin')
            self.bert_emb_size = self.bert_emb_size + 300


        if config['concat_speaker_IDs'] or config['use_unique_speaker_feature'] or config['use_unique_speaker_feature2']:
            self.speaker_emb_dim = 50
        if config['concat_speaker_IDs']:
            self.bert_emb_size = self.bert_emb_size + self.speaker_emb_dim
        if config['concat_entity_IDs'] or config['concat_entity_IDs2']:
            self.entity_emb_dim = 100
        if config['concat_entity_IDs']:
            self.bert_emb_size = self.bert_emb_size + self.entity_emb_dim

        self.span_emb_size = self.bert_emb_size * 3
        if config['use_features']:
            self.span_emb_size += config['feature_emb_size']
        if config['use_unique_speaker_feature']:
            self.span_emb_size += self.speaker_emb_dim
        self.pair_emb_size = self.span_emb_size * 3
        if config['use_metadata']:
            self.pair_emb_size += config['feature_emb_size']
        if config['use_binary_speaker_feature']:
            self.pair_emb_size += config['feature_emb_size']
        if config['use_binary_entity_feature']:
            self.pair_emb_size += config['feature_emb_size']
        if config['use_features']:
            self.pair_emb_size += config['feature_emb_size']
        if config['use_segment_distance']:
            self.pair_emb_size += config['feature_emb_size']
        if config['concat_entity_IDs2']:
            self.pair_emb_size += 3*self.entity_emb_dim*2


        self.emb_span_width = self.make_embedding(self.max_span_width) if config['use_features'] else None
        self.emb_span_width_prior = self.make_embedding(self.max_span_width) if config['use_width_prior'] else None
        self.emb_antecedent_distance_prior = self.make_embedding(10) if config['use_distance_prior'] else None
        self.emb_genre = self.make_embedding(self.num_genres)
        self.emb_same_speaker = self.make_embedding(2) if config['use_metadata'] else None
        self.emb_same_entity = self.make_embedding(2) if config['use_metadata'] else None
        self.emb_segment_distance = self.make_embedding(config['max_training_sentences']) if config[
            'use_segment_distance'] else None
        self.emb_top_antecedent_distance = self.make_embedding(10)
        self.emb_cluster_size = self.make_embedding(10) if config['higher_order'] == 'cluster_merging' else None

        self.mention_token_attn = self.make_ffnn(self.bert_emb_size, 0, output_size=1) if config[
            'model_heads'] else None
        self.span_emb_score_ffnn = self.make_ffnn(self.span_emb_size, [config['ffnn_size']] * config['ffnn_depth'],
                                                  output_size=1)
        self.span_width_score_ffnn = self.make_ffnn(config['feature_emb_size'],
                                                    [config['ffnn_size']] * config['ffnn_depth'], output_size=1) if \
            config['use_width_prior'] else None

        if config['concat_entity_IDs2']:
            self.coarse_bilinear = self.make_ffnn(self.span_emb_size+self.entity_emb_dim*2, 0, output_size=self.span_emb_size+self.entity_emb_dim*2)
            self.coref_score_ffnn = self.make_ffnn(self.pair_emb_size, [config['ffnn_size']] * config['ffnn_depth'],
                                                   output_size=1) if config['fine_grained'] else None
        else:
            self.coarse_bilinear = self.make_ffnn(self.span_emb_size, 0, output_size=self.span_emb_size)

        self.coref_score_ffnn = self.make_ffnn(self.pair_emb_size, [config['ffnn_size']] * config['ffnn_depth'],
                                                   output_size=1) if config['fine_grained'] else None
        self.antecedent_distance_score_ffnn = self.make_ffnn(config['feature_emb_size'], 0, output_size=1) if config[
            'use_distance_prior'] else None

        self.gate_ffnn = self.make_ffnn(2 * self.span_emb_size, 0, output_size=self.span_emb_size) if config[
                                                                                                          'coref_depth'] > 1 else None
        self.span_attn_ffnn = self.make_ffnn(self.span_emb_size, 0, output_size=1) if config[
                                                                                          'higher_order'] == 'span_clustering' else None
        self.cluster_score_ffnn = self.make_ffnn(3 * self.span_emb_size + config['feature_emb_size'],
                                                 [config['cluster_ffnn_size']] * config['ffnn_depth'], output_size=1) if \
            config['higher_order'] == 'cluster_merging' else None

        if config['use_unique_speaker_feature2']:
            self.span_emb_size -= self.speaker_emb_dim
            self.span_emb_score_ffnn = self.make_ffnn(self.span_emb_size, [config['ffnn_size']] * config['ffnn_depth'],
                                                           output_size=1)

        self.update_steps = 0  # Internal use for debug
        self.debug = True

    def numberbatch_emb(self, word):
        emb = self.static_word_embedding(word, self.numberbatch_dict)
        return emb

    def glove_emb(self, word):
        emb = self.static_word_embedding(word, self.glove_dict)
        return emb

    def fasttext_emb(self, word):
        emb = self.fasttext.get_word_vector(word)
        emb = torch.from_numpy(emb)
        return emb

    def static_word_embedding(self, word, emb_dict):
        if word in emb_dict:
            if isinstance(emb_dict[word], np.ndarray):
                return torch.from_numpy(emb_dict[word])
            else:
                return emb_dict[word]
        else:
            # OOV words gets assigned random 300d vector
            emb = torch.rand(300)
            if emb not in emb_dict:
                emb_dict[word] = emb
                return emb
            else:
                self.static_word_embedding(word, emb_dict)


    def embeddings_index(self, file):
        embeddings_index = dict()
        f = open(file)
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs      # Size: 300
        f.close()
        return embeddings_index


    def static_subtok_embedding(self, input_ids, emb_function):
        flat_input = torch.flatten(input_ids)
        tokens = self.tokenizer.convert_ids_to_tokens(flat_input)

        reverse_map = dict()
        pretok_sent = ""
        tok_ID = -1
        for i, tok in enumerate(tokens):
            if tok != "[PAD]":
                if tok.startswith("##"):
                    pretok_sent += tok[2:]
                    reverse_map[tok_ID] += 1
                else:
                    tok_ID += 1
                    reverse_map[tok_ID] = 1
                    pretok_sent += " " + tok
        pretok_sent = pretok_sent.split()
        static_emb = [emb_function(tok) for tok in pretok_sent]

        subtok_static_emb = []
        for i, emb in enumerate(static_emb):
            for j in range(reverse_map[i]):
                subtok_static_emb.append(emb)

        subtok_static_emb = torch.stack(subtok_static_emb, dim=0)

        return subtok_static_emb



    def add_special_speaker_tokens(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        # adding special token to separate the tokens (Bert will not split it when tokenizing)
        speaker_strings = ['[TATIANA]', '[CENNET]', '[ANNALENA]', '[JOSEF]', '[PETER]', '[FRANCOIS]']
        #speaker_strings = ['[SPK1]', '[SPK2]', '[SPK3]', '[SPK4]', '[SPK5]', '[SPK6]']
        special_tokens_dict = {'additional_special_tokens': speaker_strings}
        self.bert.resize_token_embeddings(len(self.tokenizer))
        # create token ID for each speaker
        self.speaker_id_dict = dict()
        for speaker_idx, speaker in enumerate(speaker_strings):
            speaker_token_id = self.tokenizer.convert_tokens_to_ids(speaker)
            self.speaker_id_dict[speaker_idx] = speaker_token_id

    def make_embedding(self, dict_size, std=0.02):
        emb = nn.Embedding(dict_size, self.config['feature_emb_size'])
        init.normal_(emb.weight, std=std)
        return emb

    def make_linear(self, in_features, out_features, bias=True, std=0.02):
        linear = nn.Linear(in_features, out_features, bias)
        init.normal_(linear.weight, std=std)
        if bias:
            init.zeros_(linear.bias)
        return linear

    def make_ffnn(self, feat_size, hidden_size, output_size):
        if hidden_size is None or hidden_size == 0 or hidden_size == [] or hidden_size == [0]:
            return self.make_linear(feat_size, output_size)

        if not isinstance(hidden_size, Iterable):
            hidden_size = [hidden_size]
        ffnn = [self.make_linear(feat_size, hidden_size[0]), nn.ReLU(), self.dropout]
        for i in range(1, len(hidden_size)):
            ffnn += [self.make_linear(hidden_size[i - 1], hidden_size[i]), nn.ReLU(), self.dropout]
        ffnn.append(self.make_linear(hidden_size[-1], output_size))
        return nn.Sequential(*ffnn)

    def get_params(self, named=False):
        bert_based_param, task_param = [], []
        for name, param in self.named_parameters():
            if name.startswith('bert'):
                to_add = (name, param) if named else param
                bert_based_param.append(to_add)
            else:
                to_add = (name, param) if named else param
                task_param.append(to_add)
        return bert_based_param, task_param

    def forward(self, *input):
        return self.get_predictions_and_loss(*input)

    def get_predictions_and_loss(self, input_ids, input_mask, speaker_ids, sentence_len, subtoken_map, genre, entity_ids, sentence_map,
                                 is_training=None, gold_starts=None, gold_ends=None, gold_mention_cluster_map=None,
                                 singleton_mask=None, gold_prediction_mode=False):
        """ Model and input are already on the device """
        device = self.device
        conf = self.config


        do_loss = False
        if gold_mention_cluster_map is not None:
            assert gold_starts is not None
            assert gold_ends is not None
            assert singleton_mask is not None
            if not gold_prediction_mode:
                do_loss = True

        if conf['xu_choi_speaker_feature']:
            mention_doc = self.xu_choi(input_ids, speaker_ids, input_mask)
            input_mask = input_mask.to(torch.bool)
            mention_doc = mention_doc[input_mask]
            speaker_ids = speaker_ids[input_mask]
            entity_ids = entity_ids[input_mask]
            num_words = mention_doc.shape[0]
        else:
            mention_doc = self.bert(input_ids, attention_mask=input_mask).last_hidden_state # [num seg, num max tokens, emb size]
            input_mask = input_mask.to(torch.bool)
            mention_doc = mention_doc[input_mask]
            speaker_ids = speaker_ids[input_mask]
            entity_ids = entity_ids[input_mask]
            num_words = mention_doc.shape[0]

        if conf['merge_speaker_input']:
            input_mask = input_mask.to(torch.bool)
            mention_doc = mention_doc[input_mask]
            num_words = mention_doc.shape[0]
            speaker_doc = self.bert(speaker_ids, attention_mask=input_mask).last_hidden_state
            speaker_doc = speaker_doc[input_mask]
            speaker_ids = speaker_ids[input_mask]
            mention_doc = mention_doc * 0.6 + speaker_doc * 0.4
            del speaker_doc

        if conf['concat_speaker_IDs'] or conf['use_unique_speaker_feature'] or conf['use_unique_speaker_feature2']:
            dim0 = speaker_ids.shape[0]
            speaker_embeds = torch.zeros(dim0, self.speaker_emb_dim)
            speaker2emb = dict()
            for i in range(dim0):
                speaker_id = speaker_ids[i].item()
                if speaker_id in speaker2emb:
                    speaker_embedding = speaker2emb[speaker_id]
                else:
                    speaker_embedding = torch.randn(self.speaker_emb_dim)
                    speaker2emb[speaker_id] = speaker_embedding
                speaker_embeds[i] = speaker_embedding
            speaker_embeds = speaker_embeds.to(self.device)

        if conf['concat_speaker_IDs']:
            # concat mention_doc output with speaker embeddings
            mention_doc = torch.cat((speaker_embeds, mention_doc), dim=1)

        if conf['concat_entity_IDs']:
            dim0 = entity_ids.shape[0]
            entity_embeds = torch.zeros(dim0, self.entity_emb_dim)
            entity2emb = dict()
            for i in range(dim0):
                entity_id = entity_ids[i].item()
                if entity_id in entity2emb:
                    entity_embedding = entity2emb[entity_id]
                else:
                    entity_embedding = torch.randn(self.entity_emb_dim)
                    entity2emb[entity_id] = entity_embedding
                entity_embeds[i] = entity_embedding
            entity_embeds = entity_embeds.to(self.device)
            mention_doc = torch.cat((entity_embeds, mention_doc), dim=1)

        if conf['add_numberbatch_emb']:
            numberbatch_embbeddings = self.static_subtok_embedding(input_ids, self.numberbatch_emb)
            numberbatch_embbeddings = numberbatch_embbeddings.to(self.device)
            mention_doc = torch.cat((numberbatch_embbeddings, mention_doc), dim=1)

        if conf['add_glove_emb']:
            glove_embbeddings = self.static_subtok_embedding(input_ids, self.glove_emb)
            glove_embbeddings = glove_embbeddings.to(self.device)
            mention_doc = torch.cat((glove_embbeddings, mention_doc), dim=1)

        if conf['add_fasttext_emb']:
            fasttext_embbeddings = self.static_subtok_embedding(input_ids, self.fasttext_emb)
            fasttext_embbeddings = fasttext_embbeddings.to(self.device)
            mention_doc = torch.cat((fasttext_embbeddings, mention_doc), dim=1)


        # Get candidate span
        sentence_indices = sentence_map  # [num tokens]
        if gold_prediction_mode:  # if predicting on gold, use gold starts & ends
            candidate_starts = torch.unsqueeze(gold_starts, 1)
            candidate_ends = torch.unsqueeze(gold_ends, 1)
        else:
            candidate_starts = torch.unsqueeze(torch.arange(0, num_words, device=device), 1).repeat(1,
                                                                                                    self.max_span_width)
            candidate_ends = candidate_starts + torch.arange(0, self.max_span_width, device=device)
        candidate_start_sent_idx = sentence_indices[candidate_starts]
        candidate_end_sent_idx = sentence_indices[torch.min(candidate_ends, torch.tensor(num_words - 1, device=device))]
        candidate_mask = (candidate_ends < num_words) & (candidate_start_sent_idx == candidate_end_sent_idx)
        candidate_starts, candidate_ends = candidate_starts[candidate_mask], candidate_ends[
            candidate_mask]  # [num valid candidates]
        num_candidates = candidate_starts.shape[0]

        # Get candidate labels
        if do_loss:
            same_start = (torch.unsqueeze(gold_starts, 1) == torch.unsqueeze(candidate_starts, 0))
            same_end = (torch.unsqueeze(gold_ends, 1) == torch.unsqueeze(candidate_ends, 0))
            same_span = (same_start & same_end).to(torch.long)
            candidate_labels = torch.matmul(torch.unsqueeze(gold_mention_cluster_map, 0).to(torch.float),
                                            same_span.to(torch.float))
            candidate_labels = torch.squeeze(candidate_labels.to(torch.long),
                                             0)  # [num candidates]; non-gold span has label 0
            if self.config['train_singletons']:
                candidate_labels_with_singletons = torch.matmul(
                torch.unsqueeze(gold_mention_cluster_map, 0).to(torch.float),
                same_span.to(torch.float))
                candidate_labels_with_singletons = torch.squeeze(candidate_labels_with_singletons.to(torch.long), 0)  # [num candidates]; non-gold span has label 0, singleton clusters preserved

                # mask out the singleton clusters
                gold_starts = torch.mul(gold_starts.to(torch.float), singleton_mask.to(torch.float))
                gold_ends = torch.mul(gold_ends.to(torch.float), singleton_mask.to(torch.float))
                gold_mention_cluster_map = torch.mul(gold_mention_cluster_map.to(torch.float),
                                                 singleton_mask.to(torch.float))

            same_start = (torch.unsqueeze(gold_starts, 1) == torch.unsqueeze(candidate_starts, 0))
            same_end = (torch.unsqueeze(gold_ends, 1) == torch.unsqueeze(candidate_ends, 0))
            same_span = (same_start & same_end).to(torch.long)
            candidate_labels = torch.matmul(torch.unsqueeze(gold_mention_cluster_map, 0).to(torch.float),
                                            same_span.to(torch.float))
            candidate_labels = torch.squeeze(candidate_labels.to(torch.long),
                                             0)  # [num candidates]; non-gold span has label 0

        # Get span embedding
        span_start_emb, span_end_emb = mention_doc[candidate_starts], mention_doc[candidate_ends]
        candidate_emb_list = [span_start_emb, span_end_emb]

        '''
        # plot Bert TSNE embeddings
        if gold_prediction_mode:
            mat = np.matrix([x for x in span_start_emb.tolist()])
            model = TSNE(n_components=2, random_state=80)
            low_dim_data = model.fit_transform(mat)
            tsne_df = pd.DataFrame(low_dim_data)
            ax = sns.scatterplot(data=tsne_df, x=0, y=1)
            plt.ylim(-100, 100)
            plt.xlim(-100, 100)
            ax.set_title('T-SNE SpanBERT Mention Embeddings')
            #plt.savefig(f'data/gold_AMI1_mentions_baseline.png')
            plt.show()
        '''

        if conf['use_features']:
            candidate_width_idx = candidate_ends - candidate_starts
            if gold_prediction_mode:
                emb_size = conf['feature_emb_size']
                candidate_width_emb = torch.FloatTensor([[0 for i in range(emb_size)] for id in candidate_width_idx])
            else:
                candidate_width_emb = self.emb_span_width(candidate_width_idx)
                candidate_width_emb = self.dropout(candidate_width_emb)
            candidate_emb_list.append(candidate_width_emb)
        # Use attended head or avg token
        candidate_tokens = torch.unsqueeze(torch.arange(0, num_words, device=device), 0).repeat(num_candidates, 1)
        candidate_tokens_mask = (candidate_tokens >= torch.unsqueeze(candidate_starts, 1)) & (
                candidate_tokens <= torch.unsqueeze(candidate_ends, 1))
        if conf['model_heads']:
            token_attn = torch.squeeze(self.mention_token_attn(mention_doc), 1)
        else:
            token_attn = torch.ones(num_words, dtype=torch.float, device=device)  # Use avg if no attention
        candidate_tokens_attn_raw = torch.log(candidate_tokens_mask.to(torch.float)) + torch.unsqueeze(token_attn, 0)
        candidate_tokens_attn = nn.functional.softmax(candidate_tokens_attn_raw, dim=1)
        head_attn_emb = torch.matmul(candidate_tokens_attn, mention_doc)
        candidate_emb_list.append(head_attn_emb)

        if conf['use_unique_speaker_feature'] and not conf['use_unique_speaker_feature2']:
            speaker_emb = speaker_embeds[candidate_starts]
            candidate_emb_list.append(speaker_emb)
        candidate_span_emb = torch.cat(candidate_emb_list, dim=1) # [num candidates, new emb size]

        # Get span score
        if gold_prediction_mode:  # if predicting on gold, assign same probability (1) to all candidates
            candidate_mention_scores = torch.FloatTensor([1 for i in candidate_span_emb])
        else:
            scores = self.span_emb_score_ffnn(candidate_span_emb)
            candidate_mention_scores = torch.squeeze(scores, 1)
            if conf['use_width_prior']:
                width_score = torch.squeeze(self.span_width_score_ffnn(self.emb_span_width_prior.weight), 1)
                candidate_width_score = width_score[candidate_width_idx]
                candidate_mention_scores += candidate_width_score

        # Extract top spans
        candidate_idx_sorted_by_score = torch.argsort(candidate_mention_scores, descending=True).tolist()
        candidate_starts_cpu, candidate_ends_cpu = candidate_starts.tolist(), candidate_ends.tolist()
        num_top_spans = int(min(conf['max_num_extracted_spans'], conf['top_span_ratio'] * num_words))
        selected_idx_cpu = self._extract_top_spans(candidate_idx_sorted_by_score, candidate_starts_cpu,
                                                   candidate_ends_cpu, num_top_spans)
        assert len(selected_idx_cpu) == num_top_spans
        selected_idx = torch.tensor(selected_idx_cpu, device=device)
        if candidate_starts.nelement() == 0:
            return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
        else:
            top_span_starts, top_span_ends = candidate_starts[selected_idx], candidate_ends[selected_idx]


        if conf['use_unique_speaker_feature2']:
            self.span_emb_size += self.speaker_emb_dim
            speaker_emb = speaker_embeds[candidate_starts]
            candidate_emb_list.append(speaker_emb)
            candidate_span_emb = torch.cat(candidate_emb_list, dim=1)  # [num candidates, new emb size]
            candidate_span_emb.to(device)

        if conf['concat_entity_IDs2']:
            dim0 = entity_ids.shape[0]
            entity_embeds = torch.zeros(dim0, self.entity_emb_dim)
            entity2emb = dict()
            for i in range(dim0):
                entity_id = entity_ids[i].item()
                if entity_id in entity2emb:
                    entity_embedding = entity2emb[entity_id]
                else:
                    entity_embedding = torch.randn(self.entity_emb_dim)
                    entity2emb[entity_id] = entity_embedding
                entity_embeds[i] = entity_embedding
            entity_embeds = entity_embeds.to(self.device)
            ent_mention_doc = torch.cat((entity_embeds, mention_doc), dim=1)

            span_start_emb, span_end_emb = ent_mention_doc[candidate_starts], ent_mention_doc[candidate_ends]
            candidate_emb_list[0], candidate_emb_list[1] = span_start_emb, span_end_emb
            candidate_span_emb = torch.cat(candidate_emb_list, dim=1)



        top_span_emb = candidate_span_emb[selected_idx]
        top_span_mention_scores = candidate_mention_scores[selected_idx]
        top_span_cluster_ids = candidate_labels[selected_idx] if do_loss else None


        # Coarse pruning on each mention's antecedents
        max_top_antecedents = min(num_top_spans, conf['max_top_antecedents'])
        top_span_range = torch.arange(0, num_top_spans, device=device)
        antecedent_offsets = torch.unsqueeze(top_span_range, 1) - torch.unsqueeze(top_span_range, 0)
        antecedent_mask = (antecedent_offsets >= 1)
        pairwise_mention_score_sum = torch.unsqueeze(top_span_mention_scores, 1) + torch.unsqueeze(
            top_span_mention_scores, 0)
        source_span_emb = self.dropout(self.coarse_bilinear(top_span_emb))
        target_span_emb = self.dropout(torch.transpose(top_span_emb, 0, 1))
        pairwise_coref_scores = torch.matmul(source_span_emb, target_span_emb)
        pairwise_fast_scores = pairwise_mention_score_sum + pairwise_coref_scores
        pairwise_fast_scores += torch.log(antecedent_mask.to(torch.float))
        if conf['use_distance_prior']:
            distance_score = torch.squeeze(
                self.antecedent_distance_score_ffnn(self.dropout(self.emb_antecedent_distance_prior.weight)), 1)
            bucketed_distance = util.bucket_distance(antecedent_offsets)
            antecedent_distance_score = distance_score[bucketed_distance]
            pairwise_fast_scores += antecedent_distance_score
        top_pairwise_fast_scores, top_antecedent_idx = torch.topk(pairwise_fast_scores, k=max_top_antecedents)
        top_antecedent_mask = util.batch_select(antecedent_mask, top_antecedent_idx,
                                                device)  # [num top spans, max top antecedents]
        top_antecedent_offsets = util.batch_select(antecedent_offsets, top_antecedent_idx, device)

        # Slow mention ranking
        if conf['fine_grained']:
            same_speaker_emb, genre_emb, seg_distance_emb, top_antecedent_distance_emb = None, None, None, None
            if conf['use_metadata']:
                genre_emb = self.emb_genre(genre)
                genre_emb = torch.unsqueeze(torch.unsqueeze(genre_emb, 0), 0).repeat(num_top_spans, max_top_antecedents,
                                                                                     1)
            if conf['use_binary_speaker_feature']:
                top_span_speaker_ids = speaker_ids[top_span_starts]
                top_antecedent_speaker_id = top_span_speaker_ids[top_antecedent_idx]
                same_speaker = torch.unsqueeze(top_span_speaker_ids, 1) == top_antecedent_speaker_id
                same_speaker_emb = self.emb_same_speaker(same_speaker.to(torch.long))
            if conf['use_binary_entity_feature']:
                # use entity of right-most word (possib. head?)
                top_span_entity_ids = entity_ids[top_span_ends]
                top_antecedent_entity_id = top_span_entity_ids[top_antecedent_idx]
                same_entity = torch.unsqueeze(top_span_entity_ids, 1) == top_antecedent_entity_id
                same_entity_emb = self.emb_same_entity(same_entity.to(torch.long))
            if conf['use_segment_distance']:
                num_segs, seg_len = input_ids.shape[0], input_ids.shape[1]
                token_seg_ids = torch.arange(0, num_segs, device=device).unsqueeze(1).repeat(1, seg_len)
                token_seg_ids = token_seg_ids[input_mask]
                top_span_seg_ids = token_seg_ids[top_span_starts]
                top_antecedent_seg_ids = token_seg_ids[top_span_starts[top_antecedent_idx]]
                top_antecedent_seg_distance = torch.unsqueeze(top_span_seg_ids, 1) - top_antecedent_seg_ids
                top_antecedent_seg_distance = torch.clamp(top_antecedent_seg_distance, 0,
                                                          self.config['max_training_sentences'] - 1)
                seg_distance_emb = self.emb_segment_distance(top_antecedent_seg_distance)
            if conf['use_features']:  # Antecedent distance
                top_antecedent_distance = util.bucket_distance(top_antecedent_offsets)
                top_antecedent_distance_emb = self.emb_top_antecedent_distance(top_antecedent_distance)

            for depth in range(conf['coref_depth']):
                top_antecedent_emb = top_span_emb[top_antecedent_idx]  # [num top spans, max top antecedents, emb size]
                feature_list = []
                if conf['use_metadata']:  # genre
                    feature_list.append(genre_emb)
                if conf['use_binary_speaker_feature']:
                    feature_list.append(same_speaker_emb)
                if conf['use_binary_entity_feature']:
                    feature_list.append(same_entity_emb)
                if conf['use_segment_distance']:
                    feature_list.append(seg_distance_emb)
                if conf['use_features']:  # Antecedent distance
                    feature_list.append(top_antecedent_distance_emb)
                feature_emb = torch.cat(feature_list, dim=2)
                feature_emb = self.dropout(feature_emb)
                target_emb = torch.unsqueeze(top_span_emb, 1).repeat(1, max_top_antecedents, 1)
                similarity_emb = target_emb * top_antecedent_emb
                pair_emb = torch.cat([target_emb, top_antecedent_emb, similarity_emb, feature_emb], 2)
                top_pairwise_slow_scores = torch.squeeze(self.coref_score_ffnn(pair_emb), 2)
                top_pairwise_scores = top_pairwise_slow_scores + top_pairwise_fast_scores
                if conf['higher_order'] == 'cluster_merging':
                    cluster_merging_scores = ho.cluster_merging(top_span_emb, top_antecedent_idx, top_pairwise_scores,
                                                                self.emb_cluster_size, self.cluster_score_ffnn, None,
                                                                self.dropout,
                                                                device=device, reduce=conf['cluster_reduce'],
                                                                easy_cluster_first=conf['easy_cluster_first'])
                    break
                elif depth != conf['coref_depth'] - 1:
                    if conf['higher_order'] == 'attended_antecedent':
                        refined_span_emb = ho.attended_antecedent(top_span_emb, top_antecedent_emb, top_pairwise_scores,
                                                                  device)
                    elif conf['higher_order'] == 'max_antecedent':
                        refined_span_emb = ho.max_antecedent(top_span_emb, top_antecedent_emb, top_pairwise_scores,
                                                             device)
                    elif conf['higher_order'] == 'entity_equalization':
                        refined_span_emb = ho.entity_equalization(top_span_emb, top_antecedent_emb, top_antecedent_idx,
                                                                  top_pairwise_scores, device)
                    elif conf['higher_order'] == 'span_clustering':
                        refined_span_emb = ho.span_clustering(top_span_emb, top_antecedent_idx, top_pairwise_scores,
                                                              self.span_attn_ffnn, device)

                    gate = self.gate_ffnn(torch.cat([top_span_emb, refined_span_emb], dim=1))
                    gate = torch.sigmoid(gate)
                    top_span_emb = gate * refined_span_emb + (1 - gate) * top_span_emb  # [num top spans, span emb size]
        else:
            top_pairwise_scores = top_pairwise_fast_scores  # [num top spans, max top antecedents]

        if not do_loss:
            if conf['fine_grained'] and conf['higher_order'] == 'cluster_merging':
                top_pairwise_scores += cluster_merging_scores
            top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores],
                                              dim=1)  # [num top spans, max top antecedents + 1]
            return candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends, top_antecedent_idx, top_antecedent_scores

        # Get gold labels
        top_antecedent_cluster_ids = top_span_cluster_ids[top_antecedent_idx]
        top_antecedent_cluster_ids += (top_antecedent_mask.to(
            torch.long) - 1) * 100000  # Mask id on invalid antecedents
        same_gold_cluster_indicator = (top_antecedent_cluster_ids == torch.unsqueeze(top_span_cluster_ids, 1))
        non_dummy_indicator = torch.unsqueeze(top_span_cluster_ids > 0, 1)
        pairwise_labels = same_gold_cluster_indicator & non_dummy_indicator
        dummy_antecedent_labels = torch.logical_not(pairwise_labels.any(dim=1, keepdims=True))
        top_antecedent_gold_labels = torch.cat([dummy_antecedent_labels, pairwise_labels], dim=1)

        # Get loss
        top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores], dim=1)
        if conf['loss_type'] == 'marginalized':
            log_marginalized_antecedent_scores = torch.logsumexp(
                top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float)), dim=1)
            log_norm = torch.logsumexp(top_antecedent_scores, dim=1)
            loss = torch.sum(log_norm - log_marginalized_antecedent_scores)
        elif conf['loss_type'] == 'hinge':
            top_antecedent_mask = torch.cat(
                [torch.ones(num_top_spans, 1, dtype=torch.bool, device=device), top_antecedent_mask], dim=1)
            top_antecedent_scores += torch.log(top_antecedent_mask.to(torch.float))
            highest_antecedent_scores, highest_antecedent_idx = torch.max(top_antecedent_scores, dim=1)
            gold_antecedent_scores = top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float))
            highest_gold_antecedent_scores, highest_gold_antecedent_idx = torch.max(gold_antecedent_scores, dim=1)
            slack_hinge = 1 + highest_antecedent_scores - highest_gold_antecedent_scores
            # Calculate delta
            highest_antecedent_is_gold = (highest_antecedent_idx == highest_gold_antecedent_idx)
            mistake_false_new = (highest_antecedent_idx == 0) & torch.logical_not(dummy_antecedent_labels.squeeze())
            delta = ((3 - conf['false_new_delta']) / 2) * torch.ones(num_top_spans, dtype=torch.float, device=device)
            delta -= (1 - conf['false_new_delta']) * mistake_false_new.to(torch.float)
            delta *= torch.logical_not(highest_antecedent_is_gold).to(torch.float)
            loss = torch.sum(slack_hinge * delta)

        # add mention loss with downsampling negative instances (clusters include singletons)
        if self.config['train_singletons']:
            top_span_cluster_ids_with_singletons = candidate_labels_with_singletons[selected_idx]
            gold_mention_scores = top_span_mention_scores[top_span_cluster_ids_with_singletons > 0]
            if len(gold_mention_scores) > 0 and len(top_span_cluster_ids_with_singletons) > 0:
                non_gold_mention_scores = top_span_mention_scores[top_span_cluster_ids_with_singletons == 0]
                if len(non_gold_mention_scores) > len(gold_mention_scores):
                    total_non_gold = len(non_gold_mention_scores)
                    non_gold_to_select_idx = torch.multinomial(torch.full((total_non_gold,), 1 / total_non_gold),
                                                               num_samples=len(gold_mention_scores), replacement=False)
                    non_gold_mention_scores = non_gold_mention_scores[non_gold_to_select_idx]

                loss_mention = -torch.sum(torch.log(torch.sigmoid(gold_mention_scores)))
                loss_mention += -torch.sum(torch.log(1 - torch.sigmoid(non_gold_mention_scores)))
                loss += 0.1 * loss_mention

        # add mention loss
        if conf['mention_loss_coef']:
            gold_mention_scores = top_span_mention_scores[top_span_cluster_ids > 0]
            non_gold_mention_scores = top_span_mention_scores[top_span_cluster_ids == 0]
            loss_mention = -torch.sum(torch.log(torch.sigmoid(gold_mention_scores))) * conf['mention_loss_coef']
            loss_mention += -torch.sum(torch.log(1 - torch.sigmoid(non_gold_mention_scores))) * conf[
                'mention_loss_coef']
            loss += loss_mention

        if conf['higher_order'] == 'cluster_merging':
            top_pairwise_scores += cluster_merging_scores
            top_antecedent_scores = torch.cat([torch.zeros(num_top_spans, 1, device=device), top_pairwise_scores],
                                              dim=1)
            log_marginalized_antecedent_scores2 = torch.logsumexp(
                top_antecedent_scores + torch.log(top_antecedent_gold_labels.to(torch.float)), dim=1)
            log_norm2 = torch.logsumexp(top_antecedent_scores, dim=1)  # [num top spans]
            loss_cm = torch.sum(log_norm2 - log_marginalized_antecedent_scores2)
            if conf['cluster_dloss']:
                loss += loss_cm
            else:
                loss = loss_cm

        # Debug
        if self.debug:
            if self.update_steps % 20 == 0:
                logger.info('---------debug step: %d---------' % self.update_steps)
                # logger.info('candidates: %d; antecedents: %d' % (num_candidates, max_top_antecedents))
                logger.info('spans/gold: %d/%d; ratio: %.2f' % (
                    num_top_spans, (top_span_cluster_ids > 0).sum(), (top_span_cluster_ids > 0).sum() / num_top_spans))
                if conf['mention_loss_coef']:
                    logger.info('mention loss: %.4f' % loss_mention)
                if conf['loss_type'] == 'marginalized':
                    logger.info(
                        'norm/gold: %.4f/%.4f' % (torch.sum(log_norm), torch.sum(log_marginalized_antecedent_scores)))
                else:
                    logger.info('loss: %.4f' % loss)
        self.update_steps += 1

        return [candidate_starts, candidate_ends, candidate_mention_scores, top_span_starts, top_span_ends,
                top_antecedent_idx, top_antecedent_scores], loss

    def _extract_top_spans(self, candidate_idx_sorted, candidate_starts, candidate_ends, num_top_spans):
        """ Keep top non-cross-overlapping candidates ordered by scores; compute on CPU because of loop """
        selected_candidate_idx = []
        start_to_max_end, end_to_min_start = {}, {}
        for candidate_idx in candidate_idx_sorted:
            if len(selected_candidate_idx) >= num_top_spans:
                break
            # Perform overlapping check
            span_start_idx = candidate_starts[candidate_idx]
            span_end_idx = candidate_ends[candidate_idx]
            cross_overlap = False
            for token_idx in range(span_start_idx, span_end_idx + 1):
                max_end = start_to_max_end.get(token_idx, -1)
                if token_idx > span_start_idx and max_end > span_end_idx:
                    cross_overlap = True
                    break
                min_start = end_to_min_start.get(token_idx, -1)
                if token_idx < span_end_idx and 0 <= min_start < span_start_idx:
                    cross_overlap = True
                    break
            if not cross_overlap:
                # Pass check; select idx and update dict stats
                selected_candidate_idx.append(candidate_idx)
                max_end = start_to_max_end.get(span_start_idx, -1)
                if span_end_idx > max_end:
                    start_to_max_end[span_start_idx] = span_end_idx
                min_start = end_to_min_start.get(span_end_idx, -1)
                if min_start == -1 or span_start_idx < min_start:
                    end_to_min_start[span_end_idx] = span_start_idx
        # Sort selected candidates by span idx
        selected_candidate_idx = sorted(selected_candidate_idx,
                                        key=lambda idx: (candidate_starts[idx], candidate_ends[idx]))
        if len(selected_candidate_idx) < num_top_spans:  # Padding
            if selected_candidate_idx == []:
                selected_candidate_idx += ([0] * num_top_spans)
            else:
                selected_candidate_idx += ([selected_candidate_idx[0]] * (num_top_spans - len(selected_candidate_idx)))
        return selected_candidate_idx


    def xu_choi(self, input_ids, speaker_ids, input_mask):
        try:
            self.speaker_id_dict
        except:
            self.add_special_speaker_tokens()

        combined_IDs = []
        combined_mask = []
        previous_speaker = ""
        speaker_emb_indices = []
        for j, tensor in enumerate(input_ids):
            for i, ID in enumerate(tensor):
                speaker_ID = speaker_ids[j][i]
                speaker_token_ID = self.speaker_id_dict[int(speaker_ID)]
                if speaker_ID != previous_speaker:
                    cur_idx = len(combined_IDs)
                    combined_IDs.append(speaker_token_ID)
                    combined_mask.append(input_mask[j][i])
                    speaker_emb_indices.append(cur_idx)
                combined_IDs.append(ID)
                combined_mask.append(input_mask[j][i])
                previous_speaker = speaker_ID
        combined_IDs = [combined_IDs[x:x + self.max_seg_len] for x in range(0, len(combined_IDs), self.max_seg_len)]
        combined_mask = [combined_mask[x:x + self.max_seg_len] for x in range(0, len(combined_mask), self.max_seg_len)]
        while len(combined_IDs[-1]) < self.max_seg_len:
            combined_IDs[-1].append(torch.tensor(0, ))
            combined_mask[-1].append(torch.tensor(0, device=self.device))
        combined_IDs = torch.tensor(combined_IDs).to(self.device)
        combined_mask = torch.tensor(combined_mask).to(self.device)

        mention_doc = self.bert(combined_IDs, attention_mask=combined_mask).last_hidden_state  # [num seg, num max tokens, emb size]

        # remove special tokens after first encoding
        mention_list = mention_doc.tolist()
        mention_list = [element for nestedlist in mention_list for element in nestedlist]
        speaker_emb_indices.reverse()
        for idx in speaker_emb_indices:
            del mention_list[idx]
        infused_IDs = [mention_list[x:x + 384] for x in range(0, len(mention_list), 384)]
        del infused_IDs[-1]
        mention_doc = torch.tensor(infused_IDs).to(self.device)

        return mention_doc


    def get_predicted_antecedents(self, antecedent_idx, antecedent_scores):
        """ CPU list input """
        predicted_antecedents = []
        if antecedent_scores == []:
            return predicted_antecedents
        for i, idx in enumerate(np.argmax(antecedent_scores, axis=1) - 1):
            if idx < 0:
                predicted_antecedents.append(-1)
            else:
                predicted_antecedents.append(antecedent_idx[i][idx])
        return predicted_antecedents


    def get_predicted_clusters(self, candidate_starts, candidate_ends, span_starts, span_ends, antecedent_idx,
                               antecedent_scores, candidate_mention_scores):
        """ CPU list input """
        # Get predicted antecedents
        predicted_antecedents = self.get_predicted_antecedents(antecedent_idx, antecedent_scores)

        # Get predicted clusters
        mention_to_cluster_id = {}
        predicted_clusters = []
        non_singleton_chain_mentions = set()
        for i, predicted_idx in enumerate(predicted_antecedents):
            if predicted_idx < 0:
                continue
            assert i > predicted_idx, f'span idx: {i}; antecedent idx: {predicted_idx}'
            # Check antecedent's cluster
            antecedent = (int(span_starts[predicted_idx]), int(span_ends[predicted_idx]))
            antecedent_cluster_id = mention_to_cluster_id.get(antecedent, -1)
            if antecedent_cluster_id == -1:
                antecedent_cluster_id = len(predicted_clusters)
                predicted_clusters.append([antecedent])
                mention_to_cluster_id[antecedent] = antecedent_cluster_id
            # Add mention to cluster
            mention = (int(span_starts[i]), int(span_ends[i]))
            non_singleton_chain_mentions.add(mention)
            predicted_clusters[antecedent_cluster_id].append(mention)
            mention_to_cluster_id[mention] = antecedent_cluster_id

        if self.config['train_singletons']:
            singletons_mentions = []
            # collect all the mentions with mention_score>0
            candidate_mention_abovezero_idx = np.where(np.asarray(candidate_mention_scores.cpu()) > 0)[0]
            # filter out those mentions that are not non-singletons and already exist in mention_to_cluster_id
            for i, candidate_idx in enumerate(candidate_mention_abovezero_idx):
                mention = (int(candidate_starts[int(candidate_idx)]), int(candidate_ends[int(candidate_idx)]))
                # mention = (int(span_starts[int(candidate_idx)]), int(span_ends[int(candidate_idx)]))
                if mention in non_singleton_chain_mentions:
                    continue
                singletons_mentions.append(candidate_idx)
                cluster_id = len(predicted_clusters)
                predicted_clusters.append([mention])
                mention_to_cluster_id[mention] = cluster_id
        predicted_clusters = [tuple(c) for c in predicted_clusters]

        all_men = []
        for cluster in predicted_clusters:
            for men in cluster:
                all_men.append(men)
        all_men = len(set(all_men))
        # assert len(set(non_singleton_chain_mentions)) + len(set(singletons_mentions)) == all_men

        return predicted_clusters, mention_to_cluster_id, predicted_antecedents

    def update_evaluator(self, candidate_starts, candidate_ends, candidate_mention_scores, span_starts, span_ends, antecedent_idx, antecedent_scores, gold_clusters, evaluator):
        predicted_clusters, mention_to_cluster_id, _ = self.get_predicted_clusters(candidate_starts, candidate_ends, span_starts, span_ends,
                                                                                   antecedent_idx, antecedent_scores, candidate_mention_scores)
        mention_to_predicted = {m: predicted_clusters[cluster_idx] for m, cluster_idx in mention_to_cluster_id.items()}
        gold_clusters = [tuple(tuple(m) for m in cluster) for cluster in gold_clusters]
        mention_to_gold = {m: cluster for cluster in gold_clusters for m in cluster}
        evaluator.update(predicted_clusters, gold_clusters, mention_to_predicted, mention_to_gold)
        return predicted_clusters
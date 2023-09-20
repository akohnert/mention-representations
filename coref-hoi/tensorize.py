import util
import numpy as np
import random
import os
from os.path import join
import json
import pickle
import logging
import torch

logger = logging.getLogger(__name__)


class CorefDataProcessor:
    def __init__(self, config, language='english'):
        self.config = config
        self.language = language

        self.max_seg_len = config['max_segment_len']
        self.max_training_seg = config['max_training_sentences']
        self.data_dir = config['data_dir']

        self.tokenizer = util.get_tokenizer(config['bert_tokenizer_name'])
        self.tensor_samples, self.stored_info = None, None  # For dataset samples; lazy loading

    def get_tensor_examples_from_custom_input(self, samples):
        """ For interactive samples; no caching """
        tensorizer = Tensorizer(self.config, self.tokenizer)
        tensor_samples = [tensorizer.tensorize_example(sample, False) for sample in samples]
        tensor_samples = [(doc_key, self.convert_to_torch_tensor(*tensor)) for doc_key, tensor in tensor_samples]
        return tensor_samples, tensorizer.stored_info

    def get_tensor_examples(self):
        """ For dataset samples """
        cache_path = self.get_cache_path()
        #if os.path.exists(cache_path):
        if False:
            # Load cached tensors if exists
            with open(cache_path, 'rb') as f:
                self.tensor_samples, self.stored_info = pickle.load(f)
                logger.info('Loaded tensorized examples from cache')
        else:
            # Generate tensorized samples
            self.tensor_samples = {}
            tensorizer = Tensorizer(self.config, self.tokenizer)
            paths = {
                'trn': join(self.data_dir, self.config['trn']),
                'dev': join(self.data_dir, self.config['dev']),
                'tst': join(self.data_dir, self.config['tst'])
            }
            for split, path in paths.items():
                logger.info('Tensorizing examples from %s; results will be cached)' % path)
                is_training = (split == 'trn')
                with open(path, 'r') as f:
                    samples = [json.loads(line) for line in f.readlines()]
                tensor_samples = [tensorizer.tensorize_example(sample, is_training) for sample in samples]
                self.tensor_samples[split] = [(doc_key, self.convert_to_torch_tensor(*tensor)) for doc_key, tensor
                                              in tensor_samples]
            self.stored_info = tensorizer.stored_info
            # Cache tensorized samples
            with open(cache_path, 'wb') as f:
                pickle.dump((self.tensor_samples, self.stored_info), f)
        return self.tensor_samples['trn'], self.tensor_samples['dev'], self.tensor_samples['tst']

    def get_stored_info(self):
        return self.stored_info

    @classmethod
    def convert_to_torch_tensor(cls, input_ids, input_mask, speaker_ids, sentence_len, subtoken_map, genre, entities, sentence_map,
                                is_training, gold_starts, gold_ends, gold_mention_cluster_map, singleton_mask):
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.long)
        speaker_ids = torch.tensor(speaker_ids, dtype=torch.long)
        sentence_len = torch.tensor(sentence_len, dtype=torch.long)
        subtoken_map = torch.tensor(subtoken_map, dtype=torch.long)
        genre = torch.tensor(genre, dtype=torch.long)
        entities = torch.tensor(entities, dtype=torch.long)
        sentence_map = torch.tensor(sentence_map, dtype=torch.long)
        is_training = torch.tensor(is_training, dtype=torch.bool)
        gold_starts = torch.tensor(gold_starts, dtype=torch.long)
        gold_ends = torch.tensor(gold_ends, dtype=torch.long)
        gold_mention_cluster_map = torch.tensor(gold_mention_cluster_map, dtype=torch.long)
        singleton_mask = torch.tensor(singleton_mask, dtype=torch.long)
        return input_ids, input_mask, speaker_ids, sentence_len, subtoken_map, genre, entities, sentence_map, \
               is_training, gold_starts, gold_ends, gold_mention_cluster_map, singleton_mask

    def get_cache_path(self):
        cache_path = join(self.data_dir, f'cached.tensors.{self.language}.{self.max_seg_len}.{self.max_training_seg}.bin')
        return cache_path


class Tensorizer:
    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer

        # Will be used in evaluation
        self.stored_info = {}
        self.stored_info['tokens'] = {}  # {doc_key: ...}
        self.stored_info['subtoken_maps'] = {}  # {doc_key: ...}; mapping back to tokens
        self.stored_info['gold'] = {}  # {doc_key: ...}
        self.stored_info['genre_dict'] = {genre: idx for idx, genre in enumerate(config['genres'])}

    def _tensorize_spans(self, spans):
        if len(spans) > 0:
            starts, ends = zip(*spans)
        else:
            starts, ends = [], []
        return np.array(starts), np.array(ends)

    def _tensorize_span_w_labels(self, spans, label_dict):
        if len(spans) > 0:
            starts, ends, labels = zip(*spans)
        else:
            starts, ends, labels = [], [], []
        return np.array(starts), np.array(ends), np.array([label_dict[label] for label in labels])

    def _get_speaker_dict(self, speakers):
        speaker_dict = {'UNK': 0, '[SPL]': 1}
        for speaker in speakers:
            if len(speaker_dict) > self.config['max_num_speakers']:
                pass  # 'break' to limit # speakers
            if speaker not in speaker_dict:
                speaker_dict[speaker] = len(speaker_dict)
        return speaker_dict

    def tensorize_example(self, example, is_training):
        # Mentions and clusters
        clusters = example['clusters']
        gold_mentions = sorted(tuple(mention) for mention in util.flatten(clusters))
        gold_mention_map = {mention: idx for idx, mention in enumerate(gold_mentions)}
        gold_mention_cluster_map = np.zeros(len(gold_mentions))  # 0: no cluster
        # assign 0 to all singleton clusters
        singleton_mask = np.ones(len(gold_mentions))
        for cluster_id, cluster in enumerate(clusters):
            for mention in cluster:
                gold_mention_cluster_map[gold_mention_map[tuple(mention)]] = cluster_id + 1
                if len(cluster)==1:
                    singleton_mask[gold_mention_map[tuple(mention)]] = 0
        # Speakers
        speakers = example['speakers']
        speaker_dict = self._get_speaker_dict(util.flatten(speakers))


        # Sentences/segments
        sentences = example['sentences']  # Segments
        sentence_map = example['sentence_map']
        num_words = sum([len(s) for s in sentences])
        max_sentence_len = self.config['max_segment_len']
        sentence_len = np.array([len(s) for s in sentences])

        # Bert input
        input_ids, input_mask, speaker_ids = [], [], []
        for idx, (sent_tokens, sent_speakers) in enumerate(zip(sentences, speakers)):
            sent_input_ids = self.tokenizer.convert_tokens_to_ids(sent_tokens)
            sent_input_mask = [1] * len(sent_input_ids)
            sent_speaker_ids = [speaker_dict[speaker] for speaker in sent_speakers]
            while len(sent_input_ids) < max_sentence_len:
                sent_input_ids.append(0)
                sent_input_mask.append(0)
                sent_speaker_ids.append(0)
            input_ids.append(sent_input_ids)
            input_mask.append(sent_input_mask)
            speaker_ids.append(sent_speaker_ids)
        input_ids = np.array(input_ids)
        input_mask = np.array(input_mask)
        speaker_ids = np.array(speaker_ids)
        assert num_words == np.sum(input_mask), (num_words, np.sum(input_mask))

        # Entity Types
        if 'entity_types' in example:
            entity_types = example['entity_types']
            nested = []
            for entity in entity_types:
                start = entity[0][0]
                end = entity[0][1]
                for entity2 in entity_types:
                    start2 = entity2[0][0]
                    end2 = entity2[0][1]
                    if (start >= start2) and (end <= end2) and (entity != entity2):
                        nested.append(entity2)
            entity_types = [entity for entity in entity_types if entity not in nested]
        else:
            entity_types = []

        entity_dict = dict()
        entity_map = dict()
        entity_no = 1
        for entity in entity_types:
            start = entity[0][0]
            end = entity[0][1]
            type = entity[1]
            if type not in entity_map.keys():
                entity_map[type] = entity_no
                entity_no += 1
            for i in range(start, end+1):
                entity_dict[i] = entity_map[type]


        entities = []
        for i, subtok in enumerate(input_ids.flatten()):
            if i in entity_dict:
                entities.append(entity_dict[i])
            else:
                entities.append(0)
        entities = [entities[x:x + max_sentence_len] for x in range(0, len(entities), max_sentence_len)]
        entities = np.array(entities)

        # Keep info to store
        doc_key = example['doc_key']
        self.stored_info['subtoken_maps'][doc_key] = example.get('subtoken_map', None)
        self.stored_info['gold'][doc_key] = example['clusters']
        # self.stored_info['tokens'][doc_key] = example['tokens']

        # Construct example
        subtoken_map = self.stored_info['subtoken_maps'][doc_key]
        genre = self.stored_info['genre_dict'].get(doc_key[:2], 0)
        gold_starts, gold_ends = self._tensorize_spans(gold_mentions)
        example_tensor = (input_ids, input_mask, speaker_ids, sentence_len, subtoken_map, genre, entities, sentence_map, is_training,
                          gold_starts, gold_ends, gold_mention_cluster_map, singleton_mask)
        if is_training and len(sentences) > self.config['max_training_sentences']:
            return doc_key, self.truncate_example(*example_tensor)
        else:
            return doc_key, example_tensor

    def truncate_example(self, input_ids, input_mask, speaker_ids, sentence_len, subtoken_map, genre, entities, sentence_map, is_training,
                         gold_starts, gold_ends, gold_mention_cluster_map, singleton_mask, sentence_offset=None):
        max_sentences = self.config["max_training_sentences"]
        num_sentences = input_ids.shape[0]
        assert num_sentences > max_sentences

        sent_offset = sentence_offset
        if sent_offset is None:
            sent_offset = random.randint(0, num_sentences - max_sentences)
        word_offset = sentence_len[:sent_offset].sum()
        num_words = sentence_len[sent_offset: sent_offset + max_sentences].sum()

        input_ids = input_ids[sent_offset: sent_offset + max_sentences, :]
        input_mask = input_mask[sent_offset: sent_offset + max_sentences, :]
        speaker_ids = speaker_ids[sent_offset: sent_offset + max_sentences, :]
        entities = entities[sent_offset: sent_offset + max_sentences, :]
        sentence_len = sentence_len[sent_offset: sent_offset + max_sentences]

        sentence_map = sentence_map[word_offset: word_offset + num_words]
        gold_spans = (gold_starts < word_offset + num_words) & (gold_ends >= word_offset)
        gold_starts = gold_starts[gold_spans] - word_offset
        gold_ends = gold_ends[gold_spans] - word_offset
        gold_mention_cluster_map = gold_mention_cluster_map[gold_spans]
        singleton_mask = singleton_mask[gold_spans]

        return input_ids, input_mask, speaker_ids, sentence_len, subtoken_map, genre, entities, sentence_map, \
               is_training, gold_starts, gold_ends, gold_mention_cluster_map, singleton_mask

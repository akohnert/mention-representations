import helper
import os
import json
from collections import Counter


# Convert a CoNLL-UA format file to json format file
def ua_to_json(conll_path, json_path):
    for file in os.listdir(conll_path):
        if "conll" in file.lower():
            filename = file.split(".")[0]
            helper.convert_coref_ua_to_json(UA_PATH=conll_path+f"{file}", JSON_PATH=f"{json_out}/{filename}.jsonlines", \
                                            MODEL="coref-hoi", SEGMENT_SIZE=384, TOKENIZER_NAME="bert-base-cased")

            with open(f"json/{filename}.v4_gold_conll", "w") as new_file:
                for line in file:
                    new_file.write(line)
            new_file.close()


# Convert a json format file to CoNLL-UA format file
def json_to_ua(json_file, conll_file):
    helper.convert_coref_json_to_ua(json_file, conll_file, MODEL="coref-hoi")


# Remove singletons from jsonfiles
def json_remove_singleton(jsonfile_in, jsonfile_out):
    json_out = open(jsonfile_out, "w")

    with open(jsonfile_in, "r") as f:
        d = "}"
        text = f.readlines()
        if len(text) == 1:
            text = text[0]
            s = [e + d for e in text.split(d) if e]
        else:
            s = text

        for r in s:
            r = r.strip()
            json_doc = json.loads(r)
            new_clusters = []
            singletons = 0
            for cluster in json_doc['clusters']:
                if len(cluster) == 1:
                    singletons += 1
                else:
                    new_clusters.append(cluster)
            json_doc['clusters'] = new_clusters
            json_string = json.dumps(json_doc)
            json_out.write(str(json_string) + "\n")

        json_out.close()


# Convert a predicted (output) json format file to CoNLL-UA format file
# This means the "predicted_clusters" replace "clusters"
def pred_json_to_ua(json_file, conll_file, remove_duplicates=True):
    json_pred = open("temp.jsonlines", "w")

    with open(json_file, "r") as f:
        d = "}"
        text = f.readlines()
        if len(text) == 1:
            text = text[0]
            s = [e + d for e in text.split(d) if e]
        else:
            s = text

        prev_corpus = ""
        json_pred_corpus = open("temp2.jsonlines", "w")

        for r in s:
            r = r.strip()
            json_doc = json.loads(r)

            corpus = json_doc['doc_key'].split("_")[0]

            if corpus!=prev_corpus and prev_corpus!="":
                helper.convert_coref_json_to_ua("temp2.jsonlines", conll_file+"_"+prev_corpus+".conll", MODEL="coref-hoi")
                os.remove("temp2.jsonlines")
                json_pred_corpus.close()
                json_pred_corpus = open("temp2.jsonlines", "w")

            if 'predicted_clusters' in json_doc:
                if remove_duplicates:
                    predicted_mentions_no_sing = [men for lst in json_doc['predicted_clusters'] for men in lst if len(lst)>1]
                    predicted_clusters = []
                    for lst in json_doc['predicted_clusters']:
                        if (len(lst) > 1) or (lst[0] not in predicted_mentions_no_sing):
                            predicted_clusters.append(lst)
                    json_doc['predicted_clusters'] = predicted_clusters
                json_doc['clusters'] = json_doc['predicted_clusters']
            json_string = json.dumps(json_doc)
            json_pred.write(str(json_string)+"\n")
            json_pred_corpus.write(str(json_string)+"\n")

            prev_corpus = corpus

        helper.convert_coref_json_to_ua("temp.jsonlines", conll_file, MODEL="coref-hoi")
        os.remove("temp.jsonlines")
        json_pred.close()

        helper.convert_coref_json_to_ua("temp2.jsonlines", conll_file+"_"+prev_corpus+".conll", MODEL="coref-hoi")
        os.remove("temp2.jsonlines")
        json_pred_corpus.close()

# Print some simple statistics about jsonfile
def get_stats(json_file): # Turns, chains and singletons in file per corpus
    chains = Counter()
    turns = Counter()
    singletons = Counter()

    with open(json_file, "r") as f:
        d = "}"
        text = f.readlines()
        if len(text) == 1:
            text = text[0]
            s = [e + d for e in text.split(d) if e]
        else:
            s = text

        for r in s:
            r = r.strip()
            json_doc = json.loads(r)
            corpus = json_doc["doc_key"].split("_")[0]

            doc_chains = len(json_doc["clusters"])
            chains[corpus] += doc_chains

            doc_singletons = len([cluster for cluster in json_doc["clusters"] if len(cluster)==1])
            singletons[corpus] += doc_singletons

            doc_speakers = json_doc["speakers"]
            prev_speaker = doc_speakers[0][0]
            for sent in doc_speakers:
                for speaker in sent:
                    if speaker != prev_speaker:
                        turns[corpus] += 1
                    prev_speaker = speaker

    print("Chains: ", chains)
    print("Singletons: ", singletons)
    print("Turns: ", turns)
    return chains, turns


# Combine all corpora in json format into one file
def combine_corpora(json_dir, combined_filename, mode=str()):
    combined_train = open(combined_filename, "w")

    for corpus in os.listdir(json_dir):
        if os.path.isdir(json_dir+corpus):
            for file in os.listdir(json_dir+corpus):
                filepath = os.path.join(json_dir, corpus, file)
                if os.path.isfile(filepath) and f"{mode}.english.384.jsonlines" == file:
                    f = open(filepath, "r")
                    train = f.readlines()
                    for doc in train:
                        combined_train.write(doc)

if __name__ == '__main__':
    pred_json_to_ua("data/prediction.english.384.jsonlines", "data/prediction.english.384.conllua")
    json_remove_singleton("crac_dev.384.jsonlines", "crac_dev_ns.384.jsonlines")

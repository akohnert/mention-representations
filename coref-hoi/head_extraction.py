import json
import spacy

class Headedness():
    def __init__(self):
        self.nlp = spacy.load("en_core_web_lg")

    def spacy_head(self, json_infile, json_outfile):
        json_out = open(json_outfile, 'w')
        with open(json_infile, "r") as f:
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
                tokens = json_doc["tokens"]
                clusters = json_doc["clusters"]
                subtokens = json_doc["subtoken_map"]

                heads = []
                men_sum = 0

                for cluster in clusters:
                    for men in cluster:
                        men_sum += 1
                        subtoken_start, subtoken_end = subtokens[men[0]], subtokens[men[1]]
                        token_span = " ".join(tokens[subtoken_start:subtoken_end+1])
                        span_idx = (men[0], men[1])
                        doc = self.nlp(token_span)
                        head = ""
                        for chunk in doc.noun_chunks:
                            head = chunk.root.text
                            try:
                                head_idx = (men[0] + token_span.split().index(head),
                                            men[0] + token_span.split().index(head))
                            except:
                                head = ""
                            break
                        if head == "":
                            head = token_span
                            head_idx = (men[0], men[1])
                        heads.append((span_idx, head_idx))

                assert len(heads) == men_sum

                json_doc["heads"] = heads
                json_string = json.dumps(json_doc)+"\n"
                json_out.write(json_string)
        json_out.close()



if __name__ == '__main__':
    H = Headedness()
    H.spacy_head("../mydata/gold/combined/ns_train.english.384.jsonlines", "../mydata/gold/combined/ns_heads/train.english.384.jsonlines")




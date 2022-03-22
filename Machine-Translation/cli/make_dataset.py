

## ───────────────────────────────────── ▼ ─────────────────────────────────────
# {{{                           --     INFO     --
#···············································································
# Makes translation dataset from 2 translation docs
# Prints results to outp.json

# Usage: python3 make_dataset.py <src_lang> <tgt_lang> <src_doc> <tgt_doc>>

# Example: python3 make_dataset.py en ru en_corpus ru_corpus out.json
#                                                                            }}}
## ─────────────────────────────────────────────────────────────────────────────


import sys
import json

n = len(sys.argv)

source_lang, target_lang = sys.argv[1], sys.argv[2]
source_doc, target_doc = sys.argv[3], sys.argv[4]


def load_doc(source_lang, target_lang):
    src = open(source_lang, mode='rt', encoding='utf-8')
    tgt = open(target_lang, mode='rt', encoding='utf-8')
    src_text, tgt_text = src.read(), tgt.read()
    src.close()
    tgt.close()
    src_text, tgt_text = src.read(), tgt.read()
    return src_text, tgt_text


def to_examples(src, tgt):
    return src.strip().split('\n'), tgt.strip().split('\n')


def save_examples(src, tgt):
    with open('outp.json', 'a') as f:
        cnt = 0
        while cnt < len(src) and cnt < len(tgt):
            tmpdict = {"id": str(cnt),
                       "translation":
                       {str(source_lang): src[cnt], str(target_lang): tgt[cnt]}
                       }
            json.dump(tmpdict, f, ensure_ascii=False)
            f.write('\n')
            cnt += 1


src, tgt = load_doc(source_doc, target_doc)
src_examples, tgt_examples = to_examples(src, tgt)


save_examples(src_examples, tgt_examples)

print('done')

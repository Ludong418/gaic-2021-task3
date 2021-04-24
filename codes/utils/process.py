from collections import Counter


def text_label(line, separator, label_none=None):
    line = line.strip().split(separator)
    if label_none:
        label = label_none
    else:
        label = line[-1]

    return line[0], line[1], int(label)


def int_text_label(line, separator, label_none=None):
    line = line.strip().split(separator)
    if label_none:
        label = label_none
    else:
        label = line[-1]

    # text_a = list(map(lambda x: int(x), line[0].split(' ')))
    # text_b = list(map(lambda x: int(x), line[1].split(' ')))
    text_a = line[0].split(' ')
    text_b = line[1].split(' ')

    return text_a, text_b, int(label)


def truncate_seq_pair(tokens_a, tokens_b, max_seq_len):
    """
    Truncates a sequence pair in place to the maximum length.
    (from tf-bert version)
    """
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_seq_len - 3:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

    return tokens_a, tokens_b


def truncate_seq(tokens, max_seq_len):
    return tokens[: max_seq_len]


def padding(tokens, max_seq_len, pad):
    pads = [pad] * (max_seq_len - len(tokens))
    tokens += pads

    return tokens


def count_tokens(data, min_mum):
    count = Counter()
    for d in data:
        tokens = d[0] + d[1]
        count.update(tokens)

    if min_mum:
        count = {i: j for i, j in count.items() if j >= min_mum}

    return dict(count)


def load_vocab_dict(path, encoding='utf-8'):
    token_dict = {}
    with open(path, encoding=encoding) as reader:
        for line in reader:
            token = line.split()
            token = token[0] if token else line.strip()
            token_dict[token] = len(token_dict)

    return token_dict


def write_vocab_dict(vocab, path, encoding='utf-8'):
    with open(path, 'w', encoding=encoding) as writer:
        for v in vocab:
            writer.write(str(v) + '\n')

import collections


def preprocess(text):
    if not text:
        return []
    text = str(text).lower()
    text = text.replace('.',  ' <PERIOD> ')
    text = text.replace(',',  ' <COMMA> ')
    text = text.replace('"',  ' <QUOTATION_MARK> ')
    text = text.replace(';',  ' <SEMICOLON> ')
    text = text.replace('!',  ' <EXCLAMATION_MARK> ')
    text = text.replace('?',  ' <QUESTION_MARK> ')
    text = text.replace('(',  ' <LEFT_PAREN> ')
    text = text.replace(')',  ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?',  ' <QUESTION_MARK> ')
    text = text.replace(':',  ' <COLON> ')
    words = text.split()
    return words


def create_lookup_tables(words):
    word_counts = collections.Counter(words)
    vocab = sorted(word_counts, key=lambda k: word_counts.get(k), reverse=True)
    int_to_vocab = {ii+1: word for ii, word in enumerate(vocab)}
    int_to_vocab[0] = '<PAD>'
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}
    return vocab_to_int, int_to_vocab

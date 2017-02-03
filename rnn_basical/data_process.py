import csv
import itertools
import nltk
import numpy as np

if __name__ == '__main__':
    vocabulary_size = 8000
    unknown_token = 'UNKNOWN_TOKEN'
    sentence_start_token = "SENTENCE_START"
    sentence_end_token = "SENTENCE_END"

    nltk.data.path.append('/home/aurora/hdd/nltk_data')
    with open('../data/rnn_1/reddit-comments-2015-08.csv') as f:
        reader = csv.reader(f, skipinitialspace=True)
        next(reader)
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].lower()) for x in list(reader)])
        sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in sentences]
    # print(sentences[0])
    print("Parsed %d sentences." % (len(sentences)))
    # Tokenize the sentences into words
    tokenized_sentences = [nltk.word_tokenize(sent) for sent in sentences]
    # print(tokenized_sentences)
    print(tokenized_sentences[0])
    #
    # Count the word frequencies  dict
    word_freq = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # print(type(word_freq))
    print("Found %d unique words tokens." % len(word_freq.items()))
    # Get the most common words and build index_to_word and word_to_index vectors
    vocab = word_freq.most_common(vocabulary_size - 1)
    # print(vocab)
    index_to_word = [x[0] for x in vocab]
    # print(index_to_word)
    index_to_word.append(unknown_token)
    word_to_index = dict([(w, i) for i, w in enumerate(index_to_word)])
    print("Using vocabulary size %d." % vocabulary_size)
    print("The least frequent word in our vocabulary is '%s' and appeared %d times." % (vocab[-1][0], vocab[-1][1]))

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(tokenized_sentences):
        tokenized_sentences[i] = [w if w in word_to_index else unknown_token for w in sent]

    print("\nExample sentence: '%s'" % sentences[10])
    print("\nExample sentence after Pre-processing: '%s'" % tokenized_sentences[10])

    # Create the training data
    X_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in tokenized_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in tokenized_sentences])

    np.save('../data/rnn_1/x_train', X_train)
    np.save('../data/rnn_1/y_train', y_train)

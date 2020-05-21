from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from abc import ABC, abstractmethod
import numpy as np

"""
Abstract Tokenizer
"""


class AbstractTokenizer(ABC):

    @abstractmethod
    def tokenize_corpus(self, corpus):
        raise NotImplementedError('Abstract class shall not be implemented')


"""
Default Tokenizer
"""


class DefaultTokenizer(AbstractTokenizer):

    """
    Tokenize corpus
    """

    def tokenize_corpus(self, corpus):

        # Create the tokenizer object and train on texts
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(corpus)

        # Convert text to sequences of integers
        sequences = tokenizer.texts_to_sequences(corpus)

        # Pad sequence
        """
        We need to truncate and pad the input sequences so that they are all the same length for modeling. 
        The model will learn the zero values carry no information so indeed the sequences are not the same length in terms of content, 
        but same length vectors is required to perform the computation in Keras.
        See: https://www.tensorflow.org/guide/keras/masking_and_padding
        """
        max_sequence_length = max([len(word.split()) for word in corpus])
        print('Max sequence length: {}'.format(max_sequence_length))

        x = pad_sequences(sequences,
                          maxlen=max_sequence_length,
                          padding='post')

        # Create look-up dictionaries and reverse look-ups
        word_index = tokenizer.word_index
        index_word = {value: key for key, value in word_index.items()}

        # That plus one is because of reserving padding (i.e. index zero)
        vocabulary_size = len(word_index) + 1

        print('There are {} unique words'.format(vocabulary_size))
        return np.asarray(x), max_sequence_length, word_index, index_word, vocabulary_size

# Change Log

All notable changes to this project will be documented in this file.

## 1.0.0

### Added
- Create classifiers to perform evaluation and training metrics for the project
- Create logic of encoding and decoding using face, face RGB, fashion, mnist and text file

## 1.1.0

### Added
- Create BLEU score calculator
- Create Seq2Seq model and trainer
- Create default encoder and decoder generator for the general logic
- Create supplementary cases including:
```shell
# Word -> Fashion -> Word (Use embedding)
one_layer_word_fashion_embedding
# Word -> Word -> Word (Use embedding)
one_layer_word_word_embedding
# Sentence -> Fashion -> Sentence
one_layer_sentence_fashion
# Sentence -> Fashion -> Sentence (Use word-to-word embedding)
one_layer_sentence_fashion_embedding
```

### Updated
- Update to rename text to word for the case: Word -> Fashion -> Word
- Update the constants of paths of dataset
- Update the logic in image displayer to handle the case of one row
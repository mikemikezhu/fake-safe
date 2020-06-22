from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction


class BleuScoreCalculator:

    def __init__(self, sentences):

        self.translation_dict = {}
        for sentence in sentences:

            words = sentence.split(' ')
            words = [word for word in words if word]

            translations = self.translation_dict.get(sentence)
            if not translations:
                translations = []

            if words not in translations:
                translations.append(words)
                self.translation_dict[sentence] = translations

    def calculate(self, original_corpus, translated_corpus):

        reference_corpus = [self.translation_dict[reference]
                            for reference in original_corpus]
        translated_corpus = [translation.split(
            ' ') for translation in translated_corpus]

        smoothie = SmoothingFunction().method2
        score = corpus_bleu(reference_corpus,
                            translated_corpus,
                            smoothing_function=smoothie)
        return score

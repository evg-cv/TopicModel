import gensim

from operator import add
from nltk.data import find
from sklearn.metrics.pairwise import cosine_similarity
from src.preprocess.tokenizer import TextPreprocessor
from utils.folder_file_manager import log_print
from settings import MODEL_PATH, COEFFICIENT_A, COEFFICIENT_B, COEFFICIENT_C, COEFFICIENT_D


class GTitleAligner:
    def __init__(self):
        word2vec_sample = str(find(MODEL_PATH))
        self.model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
        self.text_processor = TextPreprocessor()

    @staticmethod
    def calculate_text_feature(word_features):
        text_feature = word_features[0]
        for w_feature in word_features[1:]:
            text_feature = list(map(add, text_feature, w_feature))

        return text_feature

    def get_feature_token_words(self, text, supported_vocab=None):
        sentences = self.text_processor.tokenize_sentence(text=text)
        text_features = []
        if supported_vocab is not None:
            vocabs = supported_vocab.split(";")
            for vocab in vocabs:
                try:
                    vocab_feature = self.model[vocab.replace(" ", "")]
                    text_features.append(vocab_feature)
                except Exception as e:
                    log_print(e)
        for sentence in sentences:
            token_words = self.text_processor.tokenize_word(sample=sentence.text)
            for t_word in token_words:
                try:
                    word_feature = self.model[t_word]
                    text_features.append(word_feature)
                except Exception as e:
                    log_print(e)

        text_feature = self.calculate_text_feature(word_features=text_features)

        return text_feature

    def estimate_title_align(self, title, content, vocab):
        title_feature = self.get_feature_token_words(text=title, supported_vocab=vocab)
        content_feature = self.get_feature_token_words(text=content)
        similarity = cosine_similarity([title_feature], [content_feature])
        category_marks = round(COEFFICIENT_C - (COEFFICIENT_C /
                               (COEFFICIENT_B + (similarity[0][0] / COEFFICIENT_A) ** COEFFICIENT_D)), 1)

        return category_marks

    def calculate_similarity(self, title, content):

        title_words = []
        content_words = []
        title_sentences = self.text_processor.tokenize_sentence(text=title)
        for t_sentence in title_sentences:
            title_words = self.text_processor.tokenize_word(sample=t_sentence.text)

        content_sentences = self.text_processor.tokenize_sentence(text=content)
        for c_sentence in content_sentences:
            content_words = self.text_processor.tokenize_word(sample=c_sentence.text)

        similarity = 0

        for t_word in title_words:
            for c_word in content_words:
                try:
                    if self.model.similarity(t_word, c_word) > 0.3:
                        similarity += 1
                except Exception as e:
                    print(e)

        print(similarity)


if __name__ == '__main__':
    GTitleAligner().get_feature_token_words(text="Chol")

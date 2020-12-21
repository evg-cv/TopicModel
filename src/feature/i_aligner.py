import math

from operator import add
from sklearn.metrics.pairwise import cosine_similarity
from src.feature.extractor import FeatureExtractor
from src.preprocess.tokenizer import TextPreprocessor


class TitleAligner:
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.text_processor = TextPreprocessor()

    @staticmethod
    def calculate_text_feature(word_features):
        text_feature = word_features[0]
        for w_feature in word_features[1:]:
            text_feature = list(map(add, text_feature, w_feature))

        return text_feature

    @staticmethod
    def cosine_similarity(v1, v2):
        """compute cosine similarity of v1 to v2: (v1 dot v2)/{||v1||*||v2||)"""
        sumxx, sumxy, sumyy = 0, 0, 0
        for i in range(len(v1)):
            x = v1[i]
            y = v2[i]
            sumxx += x * x
            sumyy += y * y
            sumxy += x * y
        return sumxy / math.sqrt(sumxx * sumyy)

    def get_feature_token_words(self, text):
        sentences = self.text_processor.tokenize_sentence(text=text)
        token = ""
        for sentence in sentences:
            token += self.text_processor.tokenize_word(sample=sentence.text)
        text_features = self.feature_extractor.extract_features(text=token)
        text_feature = self.calculate_text_feature(word_features=text_features)

        return text_feature

    def estimate_title_align(self, title, content):
        title_feature = self.get_feature_token_words(text=title)
        content_feature = self.get_feature_token_words(text=content)
        similarity = cosine_similarity([title_feature], [content_feature])
        print(similarity)

        return similarity


if __name__ == '__main__':
    TitleAligner().estimate_title_align(title="", content="")

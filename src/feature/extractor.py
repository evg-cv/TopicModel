from transformers import pipeline


class FeatureExtractor:
    def __init__(self):
        self.nlp_pipe = pipeline('feature-extraction')

    def extract_features(self, text):
        features = self.nlp_pipe(text)

        return features[0][1:-1]


if __name__ == '__main__':
    FeatureExtractor().extract_features(text="This is a dog")

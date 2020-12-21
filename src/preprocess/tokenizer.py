import string
import spacy

from spacy.lang.en import English
from spacy.lang.en.stop_words import STOP_WORDS


class TextPreprocessor:
    def __init__(self):
        self.parser = English()
        self.nlp = spacy.load('en_core_web_sm')

    def tokenize_word(self, sample):
        sample = sample.replace("/", " ").replace(".", " ")

        symbols = " ".join(string.punctuation).split(" ") + ["-", "...", "”", "”", "/"]

        tokens = self.nlp(sample)
        lemmas = []
        for tok in tokens:
            lemmas.append(tok.lemma_.lower().strip() if tok.lemma_ != "-PRON-" else tok.lower_)

        tokens = lemmas
        tokens = [tok for tok in tokens if tok not in STOP_WORDS]
        tokens = [tok for tok in tokens if tok not in symbols]
        tokens = [tok for tok in tokens if tok != ""]
        # token_str = ""
        # for tk in tokens:
        #     token_str += tk + " "

        return tokens

    def tokenize_sentence(self, text):
        about_txt = self.nlp(text)
        sentences = list(about_txt.sents)

        return sentences


if __name__ == '__main__':
    text_processor = TextPreprocessor()
    token_sentences = text_processor.tokenize_sentence("In practice, the main difference is that GloVe embeddings "
                                                           "work "
                                                           "better on some "
                                                           "data sets, while word2vec embeddings work better on "
                                                       "others. "
                                                           "They both do very "
                                                           "well at capturing the semantics of analogy, and that "
                                                       "takes us, "
                                                           "it turns out, "
                                                           "a very long way toward lexical semantics in general.")
    for t_sentence in token_sentences:
        token_words = text_processor.tokenize_word(sample=t_sentence.text)
        print(token_words)

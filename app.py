import argparse

from src.feature.g_aligner import GTitleAligner

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--title', action='store', help='A title string')
    parser.add_argument('--content', action='store', help='A content string')
    parser.add_argument('--vocab', action='store', help='A supported vocabularies')
    title = parser.parse_args().title
    content = parser.parse_args().content
    vocab = parser.parse_args().vocab

    category_marks = GTitleAligner().estimate_title_align(title=title, content=content, vocab=vocab)

    print(f"Title Align:{category_marks}")

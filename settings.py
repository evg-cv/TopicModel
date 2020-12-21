import os

CUR_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(CUR_DIR, 'utils', 'model', 'pruned.word2vec.txt')

COEFFICIENT_A = 0.63
COEFFICIENT_B = 1.0
COEFFICIENT_C = 5.0
COEFFICIENT_D = 7.0

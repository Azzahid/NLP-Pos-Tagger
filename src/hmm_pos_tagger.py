import dill
with open('my_tagger.dill', 'rb') as f:
    hmm_tagger = dill.load(f)

from nltk.tag import hmm
from sklearn.externals import joblib
from nltk.tag.hmm  import HiddenMarkovModelTagger, HiddenMarkovModelTrainer
from nltk.probability import LidstoneProbDist

sentence = raw_input("Kalimat yang ingin diambil pos taggernya: \n")

words = sentence.split()

tagged_sentences = hmm_tagger.tag(words)

print tagged_sentences

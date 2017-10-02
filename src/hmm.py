import re
from nltk.tag import hmm
from sklearn.externals import joblib
from nltk.tag.hmm  import HiddenMarkovModelTagger, HiddenMarkovModelTrainer
from nltk.probability import LidstoneProbDist

def tokenize_conllu_file( file_dir ):
    file = open(file_dir, 'r')
    line = file.readline()
    sentences = []
    sentence = []
    for line in file:
        if line[0] not in ['#','\n']:
            conllu_array = re.split(r'\t+', line)
            if conllu_array[0] == '1':
                if sentence :
                    sentences.append(sentence)
                sentence = []
            word = [conllu_array[1], conllu_array[3]]
            sentence.append((conllu_array[1], conllu_array[3]))

    return sentences


sentences = tokenize_conllu_file('../en-ud-dev.conllu')
cutoff = int(.9 * len(sentences))
training_sentences = sentences[:cutoff]
test_sentences = sentences[cutoff:]

print('Training Sentences : %d ' % (len(training_sentences)))
print('Testing Sentences : %d ' % (len(test_sentences)))


print 'Training Start'
trainer = HiddenMarkovModelTrainer()
tagger = trainer.train_supervised(training_sentences, estimator=lambda fd, bins: LidstoneProbDist(fd, 0.1, bins))
print 'Training Completed'

print 'Testing Start'
tagger.test(test_sentences, verbose=False)
print 'Testing Completed'


import dill
with open('my_tagger.dill', 'wb') as f:
    dill.dump(tagger, f)

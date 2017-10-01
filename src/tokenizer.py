import re
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
            sentence.append(word)

    return sentences

def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index][0],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0][0].upper() == sentence[index][0][0],
        'is_all_caps': sentence[index][0].upper() == sentence[index][0],
        'is_all_lower': sentence[index][0].lower() == sentence[index][0],
        'prefix-1': sentence[index][0][0],
        'prefix-2': sentence[index][0][:2],
        'prefix-3': sentence[index][0][:3],
        'suffix-1': sentence[index][0][-1],
        'suffix-2': sentence[index][0][-2:],
        'suffix-3': sentence[index][0][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1][0],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1][0],
        'has_hyphen': '-' in sentence[index][0],
        'is_numeric': sentence[index][0].isdigit(),
        'capitals_inside': sentence[index][0][1:].lower() != sentence[index][0][1:]
    }

def transform_to_dataset(sentences):
    X, Y = [], []
    for sentence in sentences:
        for index in range(len(sentence)):
            X.append(features(sentence, index))
            Y.append(sentence[index][1])

    return X, Y


sentences = tokenize_conllu_file('../en-ud-dev.conllu')
cutoff = int(.75 * len(sentences))
training_sentences = sentences[:cutoff]
test_sentences = sentences[cutoff:]

X, y = transform_to_dataset(training_sentences)

print len(X)
# for i in range(0, len(X)):
#     print X[i]
#     print y[i]
#     print '\n\n'

from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline

clf = Pipeline([
    ('vectorizer', DictVectorizer(sparse=False)),
    ('classifier', DecisionTreeClassifier(criterion='entropy'))
])

clf.fit(X, y)   # Use only the first 10K samples if you're running it multiple times. It takes a fair bit :)

print 'Training completed'

X_test, y_test = transform_to_dataset(test_sentences)

print "Accuracy:", clf.score(X_test, y_test)

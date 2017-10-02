from sklearn.externals import joblib
import re

def tokenize(sentence_input):
    sentences = []
    sentence = []
    data = sentence_input.split()
    
    for word in data:
        sentence.append(word)
    
    sentences.append(sentence)
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
        'prev_pos': '' if index == 0 else sentence[index - 1][1],
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

clf = joblib.load('DT-model.pkl')

sentence = raw_input("> Masukan kalimat yang ingin dilakukan POS TAGGER: ")

sentence_token = tokenize(sentence)

X, Y = transform_to_dataset(sentence_token)

print(clf.predict(X))



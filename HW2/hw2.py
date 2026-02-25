import re
import sys

import nltk
import numpy as np
from sklearn.linear_model import LogisticRegression


negation_words = set(['not', 'no', 'never', 'nor', 'cannot'])
negation_enders = set(['but', 'however', 'nevertheless', 'nonetheless'])
sentence_enders = set(['.', '?', '!', ';'])


# Loads a training or test corpus
# corpus_path is a string
# Returns a list of (string, int) tuples
def load_corpus(corpus_path):
    data = []
    with open(corpus_path, "r", encoding="latin-1") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            text, label = line.split("\t")
            snippet = text.split()
            label = int(label)
            data.append((snippet, label))
    return data


# Checks whether or not a word is a negation word
# word is a string
# Returns a boolean
def is_negation(word):
    if word in negation_words:
        return True
    if word.endswith("n't"):
        return True
    return False


# Modifies a snippet to add negation tagging
# snippet is a list of strings
# Returns a list of strings
def tag_negation(snippet):
    tagged = []
    pos_tags = nltk.pos_tag(snippet)
    negating = False

    i = 0
    while i < len(snippet):
        word = snippet[i]
        pos = pos_tags[i][1]

        if is_negation(word):
            # special case: "not only"
            if word == "not" and i + 1 < len(snippet) and snippet[i+1] == "only":
                tagged.append(word)
                i += 1
                continue
            negating = True
            tagged.append(word)
            i += 1
            continue

        if negating:
            # stop conditions
            if word in sentence_enders or word in negation_enders or pos in ["JJR", "RBR"]:
                negating = False
                tagged.append(word)
            else:
                tagged.append("NOT " + word)
        else:
            tagged.append(word)

        i += 1

    return tagged


# Assigns to each unigram an index in the feature vector
# corpus is a list of tuples (snippet, label)
# Returns a dictionary {word: index}
def get_feature_dictionary(corpus):
    feature_dict = {}
    idx = 0
    for snippet, _ in corpus:
        for word in snippet:
            if word not in feature_dict:
                feature_dict[word] = idx
                idx += 1
    return feature_dict
    

# Converts a snippet into a feature vector
# snippet is a list of tuples (word, pos_tag)
# feature_dict is a dictionary {word: index}
# Returns a Numpy array
def vectorize_snippet(snippet, feature_dict):
    vec = np.zeros(len(feature_dict))
    for word in snippet:
        if word in feature_dict:
            vec[feature_dict[word]] += 1
    return vec


# Trains a classification model (in-place)
# corpus is a list of tuples (snippet, label)
# feature_dict is a dictionary {word: label}
# Returns a tuple (X, Y) where X and Y are Numpy arrays
def vectorize_corpus(corpus, feature_dict):
    n = len(corpus)
    d = len(feature_dict)
    X = np.zeros((n, d))
    Y = np.zeros(n)

    for i, (snippet, label) in enumerate(corpus):
        X[i] = vectorize_snippet(snippet, feature_dict)
        Y[i] = label

    return X, Y


# Performs min-max normalization (in-place)
# X is a Numpy array
# No return value
def normalize(X):
    X = X.astype(float)
    for j in range(X.shape[1]):
        col_min = X[:, j].min()
        col_max = X[:, j].max()
        if col_max > col_min:
            X[:, j] = (X[:, j] - col_min) / (col_max - col_min)
        else:
            X[:, j] = 0.0
    return X


# Trains a model on a training corpus
# corpus_path is a string
# Returns a LogisticRegression
def train(corpus_path):
    corpus = load_corpus(corpus_path)
    corpus = [(tag_negation(s), y) for s, y in corpus]

    feature_dict = get_feature_dictionary(corpus)
    X, Y = vectorize_corpus(corpus, feature_dict)
    X = normalize(X)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, Y)

    return model, feature_dict


# Calculate precision, recall, and F-measure
# Y_pred is a Numpy array
# Y_test is a Numpy array
# Returns a tuple of floats
def evaluate_predictions(Y_pred, Y_test):
    tp = fp = fn = 0

    for yp, yt in zip(Y_pred, Y_test):
        if yt == 1 and yp == 1:
            tp += 1
        elif yt == 0 and yp == 1:
            fp += 1
        elif yt == 1 and yp == 0:
            fn += 1

    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    fmeasure = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0

    return precision, recall, fmeasure


# Evaluates a model on a test corpus and prints the results
# model is a LogisticRegression
# corpus_path is a string
# Returns a tuple of floats
def test(model, feature_dict, corpus_path):
    corpus = load_corpus(corpus_path)
    corpus = [(tag_negation(s), y) for s, y in corpus]

    X_test, Y_test = vectorize_corpus(corpus, feature_dict)
    X_test = normalize(X_test)

    Y_pred = model.predict(X_test)
    return evaluate_predictions(Y_pred, Y_test)


# Selects the top k highest-weight features of a logistic regression model
# logreg_model is a trained LogisticRegression
# feature_dict is a dictionary {word: index}
# k is an int
def get_top_features(logreg_model, feature_dict, k=1):
    # Get weight vector (shape: [num_features])
    weights = logreg_model.coef_[0]

    # Reverse feature_dict: index -> word
    idx_to_word = {idx: word for word, idx in feature_dict.items()}

    # Sort feature indices by absolute weight (descending)
    sorted_indices = sorted(range(len(weights)), key=lambda i: abs(weights[i]), reverse=True)

    # Take top-k features
    top_features = []
    for i in sorted_indices[:k]:
        top_features.append((idx_to_word[i], weights[i]))

    return top_features


def main(args):
    model, feature_dict = train('train.txt')

    print(test(model, feature_dict, 'test.txt'))

    weights = get_top_features(model, feature_dict)
    for weight in weights:
        print(weight)
    
if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

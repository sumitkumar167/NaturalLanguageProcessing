import sys

import nltk
from nltk.corpus import brown
import numpy
from scipy.sparse import csr_matrix
from sklearn.linear_model import LogisticRegression

# Load the Brown corpus with Universal Dependencies tags
# proportion is a float
# Returns a tuple of lists (sents, tags)
def load_training_corpus(proportion=1.0):
    brown_sentences = brown.tagged_sents(tagset='universal')
    num_used = int(proportion * len(brown_sentences))

    corpus_sents, corpus_tags = [None] * num_used, [None] * num_used
    for i in range(num_used):
        corpus_sents[i], corpus_tags[i] = zip(*brown_sentences[i])
    return (corpus_sents, corpus_tags)


# Generate word n-gram features
# words is a list of strings
# i is an int
# Returns a list of strings
def get_ngram_features(words, i):
    # Get surrounding words with boundary checks
    prev_word = words[i-1] if i > 0 else '<s>'
    next_word = words[i+1] if i < len(words) - 1 else '</s>'
    prev_prev_word = words[i-2] if i > 1 else '<s>'
    next_next_word = words[i+2] if i < len(words) - 2 else '</s>'
    
    # Build the list of features
    features = [
        f'prevbigram-{prev_word}',
        f'nextbigram-{next_word}',
        f'prevskip-{prev_prev_word}',
        f'nextskip-{next_next_word}',
        f'prevtrigram-{prev_word}-{prev_prev_word}',
        f'nexttrigram-{next_word}-{next_next_word}',
        f'centertrigram-{prev_word}-{next_word}'
    ]
    
    return features


# Generate word-based features
# word is a string
# returns a list of strings
def get_word_features(word):
    features = []
    
    # word feature
    features.append(f'word-{word}')
    
    # capital feature
    if word and word[0].isupper():
        features.append('capital')
    
    # allcaps feature
    if word and word.isupper() and any(c.isalpha() for c in word):
        features.append('allcaps')
    
    # wordshape feature
    wordshape = ''
    for char in word:
        if char.islower():
            wordshape += 'x'
        elif char.isupper():
            wordshape += 'X'
        elif char.isdigit():
            wordshape += 'd'
        else:
            wordshape += char
    features.append(f'wordshape-{wordshape}')
    
    # short-wordshape feature
    short_wordshape = ''
    prev_type = None
    for char in word:
        if char.islower():
            char_type = 'x'
        elif char.isupper():
            char_type = 'X'
        elif char.isdigit():
            char_type = 'd'
        else:
            char_type = 'other'
        
        if char_type != prev_type:
            short_wordshape += char_type
            prev_type = char_type
    features.append(f'short-wordshape-{short_wordshape}')
    
    # number feature
    if any(char.isdigit() for char in word):
        features.append('number')
    
    # hyphen feature
    if '-' in word:
        features.append('hyphen')
    
    # prefix features
    for j in range(1, 5):
        if j <= len(word):
            prefix = word[:j]
            features.append(f'prefix{j}-{prefix}')
    
    # suffix features
    for j in range(1, 5):
        if j <= len(word):
            suffix = word[-j:]
            features.append(f'suffix{j}-{suffix}')
    
    return features


# Wrapper function for get_ngram_features and get_word_features
# words is a list of strings
# i is an int
# prevtag is a string
# Returns a list of strings
def get_features(words, i, prevtag):
    # Get the current word
    current_word = words[i]
    
    # Get ngram features and word features
    ngram_features = get_ngram_features(words, i)
    word_features = get_word_features(current_word)
    
    # Combine into a single list
    features = ngram_features + word_features
    
    # Add tag bigram feature
    features.append(f'tagbigram-{prevtag}')
    
    # Convert all features to lowercase except wordshape and short-wordshape
    processed_features = []
    for feature in features:
        if 'wordshape-' in feature or 'short-wordshape-' in feature:
            processed_features.append(feature)
        else:
            processed_features.append(feature.lower())
    
    return processed_features


# Remove features that occur fewer than a given threshold number of time
# corpus_features is a list of lists, where each sublist corresponds to a sentence and has elements that are lists of strings (feature names)
# threshold is an int
# Returns a tuple (corpus_features, common_features)
def remove_rare_features(corpus_features, threshold=5):
    # Count feature occurrences
    feature_counts = {}
    for sentence_features in corpus_features:
        for word_features in sentence_features:
            for feature in word_features:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
    
    # Identify common features
    common_features = set()
    for feature, count in feature_counts.items():
        if count >= threshold:
            common_features.add(feature)
    
    # Create a new corpus_features with rare features removed
    filtered_corpus_features = []
    for sentence_features in corpus_features:
        filtered_sentence = []
        for word_features in sentence_features:
            filtered_word_features = [f for f in word_features if f in common_features]
            filtered_sentence.append(filtered_word_features)
        filtered_corpus_features.append(filtered_sentence)
    
    return (filtered_corpus_features, common_features)


# Build feature and tag dictionaries
# common_features is a set of strings
# corpus_tags is a list of lists of strings (tags)
# Returns a tuple (feature_dict, tag_dict)
def get_feature_and_label_dictionaries(common_features, corpus_tags):
    # Create feature dictionary
    feature_dict = {}
    for idx, feature in enumerate(sorted(common_features)):
        feature_dict[feature] = idx
    
    # Collect all unique tags from corpus_tags
    tags = set()
    for sentence_tags in corpus_tags:
        for tag in sentence_tags:
            tags.add(tag)
    
    # Create tag dictionary
    tag_dict = {}
    for idx, tag in enumerate(sorted(tags)):
        tag_dict[tag] = idx
    
    return (feature_dict, tag_dict)

# Build the label vector Y
# corpus_tags is a list of lists of strings (tags)
# tag_dict is a dictionary {string: int}
# Returns a Numpy array
def build_Y(corpus_tags, tag_dict):
    y_list = []
    for sentence_tags in corpus_tags:
        for tag in sentence_tags:
            y_list.append(tag_dict[tag])
    
    return numpy.array(y_list)

# Build a sparse input matrix X
# corpus_features is a list of lists, where each sublist corresponds to a sentence and has elements that are lists of strings (feature names)
# feature_dict is a dictionary {string: int}
# Returns a Scipy.sparse csr_matrix
def build_X(corpus_features, feature_dict):
    rows = []
    cols = []
    example_idx = 0
    
    for sentence_features in corpus_features:
        for word_features in sentence_features:
            for feature in word_features:
                if feature in feature_dict:
                    rows.append(example_idx)
                    cols.append(feature_dict[feature])
            example_idx += 1
    
    values = [1] * len(rows)
    
    rows_array = numpy.array(rows)
    cols_array = numpy.array(cols)
    values_array = numpy.array(values)
    
    # Specify the shape to ensure correct number of columns
    return csr_matrix((values_array, (rows_array, cols_array)), shape=(example_idx, len(feature_dict)))


# Train an MEMM tagger on the Brown corpus
# proportion is a float
# Returns a tuple (model, feature_dict, tag_dict)
def train(proportion=1.0):
    # Load the training corpus
    corpus_sents, corpus_tags = load_training_corpus(proportion)
    
    # Build corpus_features
    corpus_features = []
    for sent_idx, (sentence, tags) in enumerate(zip(corpus_sents, corpus_tags)):
        sentence_features = []
        for word_idx, word in enumerate(sentence):
            # Get the previous tag (use '<S>' for i=0)
            prevtag = '<S>' if word_idx == 0 else tags[word_idx - 1]
            # Get features for this word
            word_features = get_features(sentence, word_idx, prevtag)
            sentence_features.append(word_features)
        corpus_features.append(sentence_features)
    
    # Remove rare features
    corpus_features, common_features = remove_rare_features(corpus_features)
    
    # Build feature and tag dictionaries
    feature_dict, tag_dict = get_feature_and_label_dictionaries(common_features, corpus_tags)
    
    # Build X and Y
    X_train = build_X(corpus_features, feature_dict)
    Y_train = build_Y(corpus_tags, tag_dict)
    
    # Instantiate and train the model
    model = LogisticRegression(class_weight='balanced', solver='saga')
    model.fit(X_train, Y_train)
    
    return (model, feature_dict, tag_dict)



# Load the test set
# corpus_path is a string
# Returns a list of lists of strings (words)
def load_test_corpus(corpus_path):
    with open(corpus_path) as inf:
        lines = [line.strip().split() for line in inf]
    return [line for line in lines if len(line) > 0]


# Predict tags for a test sentence
# test_sent is a list containing a single list of strings
# model is a trained LogisticRegression
# feature_dict is a dictionary {string: int}
# reverse_tag_dict is a dictionary {int: string}
# Returns a tuple (Y_start, Y_pred)
def get_predictions(test_sent, model, feature_dict, reverse_tag_dict):
    sentence = test_sent[0]
    n = len(sentence)
    T = len(reverse_tag_dict)
    
    # Create Y_pred of size (n-1) x T
    Y_pred = numpy.empty((n - 1, T, T))
    
    # Process words 1 to n-1 (skip first word)
    for i in range(1, n):
        features = []
        # Generate features for all possible previous tags
        for prevtag_idx in range(T):
            prevtag = reverse_tag_dict[prevtag_idx]
            word_features = get_features(sentence, i, prevtag)
            features.append(word_features)
        
        # Build input matrix X (size T x F)
        X = build_X([features], feature_dict)
        
        # Get log probability predictions
        log_probs = model.predict_log_proba(X)
        
        # Copy to Y_pred
        Y_pred[i - 1] = log_probs
    
    # Process first word w0
    prevtag = '<S>'
    word_features = get_features(sentence, 0, prevtag)
    X_start = build_X([[word_features]], feature_dict)
    Y_start = model.predict_log_proba(X_start)
    
    return (Y_start, Y_pred)


# Perform Viterbi decoding using predicted log probabilities
# Y_start is a Numpy array of size (1, T)
# Y_pred is a Numpy array of size (n-1, T, T)
# Returns a list of strings (tags)
def viterbi(Y_start, Y_pred):
    # Get dimensions
    n = Y_pred.shape[0] + 1  # number of words
    T = Y_start.shape[1]     # number of tags
    
    # Create DP tables
    V = numpy.empty((n, T))
    BP = numpy.empty((n, T), dtype=int)
    
    # Base case: initialize V[0] with Y_start
    V[0] = Y_start[0]
    
    # Fill in the rest of V and BP
    for i in range(1, n):
        for j in range(T):
            # Compute V[i][j] = max over k of (V[i-1][k] + Y_pred[i-1][k][j])
            scores = V[i - 1] + Y_pred[i - 1, :, j]
            V[i][j] = numpy.max(scores)
            BP[i][j] = numpy.argmax(scores)
    
    # Backtrack to find the best path
    path = []
    # Start from the last position
    current_tag = numpy.argmax(V[n - 1])
    path.append(current_tag)
    
    # Trace back through backpointers
    for i in range(n - 1, 0, -1):
        current_tag = BP[i][current_tag]
        path.append(current_tag)
    
    # Reverse the path to get it in forward order
    path.reverse()
    
    return path


# Predict tags for a test corpus using a trained model
# corpus_path is a string
# model is a trained LogisticRegression
# feature_dict is a dictionary {string: int}
# tag_dict is a dictionary {string: int}
# Returns a list of lists of strings (tags)
def predict(corpus_path, model, feature_dict, tag_dict):
    # Load test data
    test_corpus = load_test_corpus(corpus_path)
    
    # Create reverse tag dictionary
    reverse_tag_dict = {v: k for k, v in tag_dict.items()}
    
    # Predict tags for each sentence
    predictions = []
    for test_sent in test_corpus:
        # Wrap the sentence in a list to match expected format
        test_sent_wrapped = [test_sent]
        
        # Get predictions
        Y_start, Y_pred = get_predictions(test_sent_wrapped, model, feature_dict, reverse_tag_dict)
        
        # Use Viterbi to get best tag sequence
        tag_indices = viterbi(Y_start, Y_pred)
        
        # Convert indices to tags
        tags = [reverse_tag_dict[idx] for idx in tag_indices]
        predictions.append(tags)
    
    return predictions


def main(args):
    model, feature_dict, tag_dict = train(0.25)

    predictions = predict('test.txt', model, feature_dict, tag_dict)
    for test_sent in predictions:
        print(test_sent)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))

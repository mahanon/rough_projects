#######
#######
# Core classifier code script.

import string

from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import wordpunct_tokenize
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report as clsr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cross_validation import train_test_split as tts

def identity(arg):
    """
    Simple identity function works as a passthrough.
    """
    return arg


#######
#######
# NTLKPreprocessor
# Modified in large part from Benjamin Bengfort public code.
    
class NLTKPreprocessor(BaseEstimator, TransformerMixin):

    def __init__(self, stopwords=None, punct=None, lower=True, strip=True):
        self.lower      = lower
        self.strip      = strip
        self.stopwords  = stopwords or set(sw.words('english'))
        self.stopwords = [self.stopwords,'theblaze','spectator','national review','infowars','dailywire','western journal','huffpost','new repblic', 'thinkprogress','vox','salon','politicususa']
        self.punct      = punct or set(string.punctuation)
        self.lemmatizer = WordNetLemmatizer()

    def fit(self, X, y=None):
        return self

    def inverse_transform(self, X):
        return [" ".join(doc) for doc in X]

    def transform(self, X):
        return [list(self.tokenize(doc)) for doc in X]

    def tokenize(self, document):
        # Break the document into sentences
        # ~ Problem here!
        #print(document[:20])
        for sent in sent_tokenize(document):
            # Break the sentence into part of speech tagged tokens
            for token, tag in pos_tag(wordpunct_tokenize(sent)):
                # Apply preprocessing to the token
                token = token.lower() if self.lower else token
                token = token.strip() if self.strip else token
                token = token.strip('_') if self.strip else token
                token = token.strip('*') if self.strip else token

                # If stopword, ignore token and continue
                if token in self.stopwords:
                    continue

                # If punctuation, ignore token and continue
                if all(char in self.punct for char in token):
                    continue

                # Lemmatize the token and yield
                lemma = self.lemmatize(token, tag)
                yield lemma

    def lemmatize(self, token, tag):
        tag = {'N': wn.NOUN,'V': wn.VERB,'R': wn.ADV,'J': wn.ADJ}.get(tag[0], wn.NOUN)
        
        return self.lemmatizer.lemmatize(token, tag)

#######
#######
# Core processing, classifier code.
# Very basic, for checking several classifiers (manually substituted).
# Grid search and whatnot must be added for MLP model, but goal is to
# switch over to tensorflow anyway.

def build_and_evaluate(text, leanings, classifier=SGDClassifier, verbose=True):

    def build(classifier, X, y=None):
        if isinstance(classifier, type):
            classifier = classifier()
        model = Pipeline([('preprocessor', NLTKPreprocessor()), ('vectorizer', TfidfVectorizer(tokenizer=identity, preprocessor=None, lowercase=False)),('classifier', classifier),])
        model.fit(X, y)
        return model

    # Label encode the targets
    labels = LabelEncoder()
    leanings = labels.fit_transform(leanings)
 
    # Build model on training data.
    text_train, text_test, leanings_train, leanings_test = tts(text, leanings, test_size=0.2)
    #build(classifier, text_train, leanings_train)
    
    model = build(classifier, text_train, leanings_train)

    leanings_pred = model.predict(text_test)
    leanings_pred_prob = model.predict_proba(text_test)
    print(clsr(leanings_test, leanings_pred, target_names=labels.classes_))

    # Build model on all data.
    model = build(classifier, text, leanings)
    model.labels_ = labels

    return leanings_test,leanings_pred_prob,model

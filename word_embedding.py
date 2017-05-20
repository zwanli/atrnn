import os
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
import tf_glove

corpusdir= 'abstract/'
corpus = PlaintextCorpusReader(corpusdir, '.*')

model = tf_glove.GloVeModel(embedding_size=200, context_size=10, min_occurrences=25,
                            learning_rate=0.05, batch_size=512)
model.fit_to_corpus(corpus.sents())
model.train(num_epochs=50, log_dir="log/example", summary_batch_interval=1000)

import os
from nltk.corpus.reader.plaintext import PlaintextCorpusReader
import tf_glove

corpusdir= 'abstract/'
corpus = PlaintextCorpusReader(corpusdir, '.*')

model = tf_glove.GloVeModel(embedding_size=200, context_size=10, min_occurrences=25,
                            learning_rate=0.05, batch_size=512)
model.fit_to_corpus(corpus.sents())
model.train(num_epochs=50, log_dir="log/example", summary_batch_interval=1000)
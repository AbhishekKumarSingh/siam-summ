# import statements
from keras.layers.embeddings import Embedding
from keras.layers.wrappers import TimeDistributed
from keras.layers.convolutional import Convolution1D
from keras.layers.pooling import MaxPooling1D, GlobalMaxPooling1D
from keras.engine.topology import Merge
from keras.layers.core import Dropout, RepeatVector, Flatten, TimeDistributedDense
from keras.layers import LSTM, Input, Dense, Activation, Lambda
from keras.models import Model, Sequential
from keras import backend as K
from IPython.display import SVG
from keras.utils.visualize_util import model_to_dot
from keras.optimizers import RMSprop, Adam
import numpy as np

np.random.seed(1337) # for reproducibility

def load_bin_vecs(fname):
    """
    Loads 300 x 1 word vecs from Google (Mikolov) word2vec
    """
    word_vecs = {}
    with open(fname, "rb") as f:
        header = f.readline()
        vocab_size, layer1size = map(int, header.split())
        # use only 150 dimension vectors
        layer1size = 150
        print vocab_size, layer1size
        binary_len = np.dtype('float32').itemsize * layer1size
        for line in xrange(vocab_size):
            word = []
            while True:
                ch = f.read(1)
                if ch == ' ':
                    word = ''.join(word)
                    break
                if ch != '\n':
                    word.append(ch)
            word_vecs[word] = np.fromstring(f.read(binary_len), dtype='float32')
    return word_vecs

fname = "/home/abhishek/workspace/reg_summ/RegressionSummarization/WordVectors/google_w2v.bin"
word_vecs = load_bin_vecs(fname)
# print len(word_vecs["the"])
#print word_vecs["cat"]


import cPickle

word_id = {}
id_word = {}

filename = "final_word_vocab"

with open(filename,'rb') as fp:
    word_id = cPickle.load(fp)
    id_word = cPickle.load(fp)

def vectorize_document(docMat):
    """takes document matrix return vectorize document matrix
       <UNK> token -> 0 for unknown words
    """
    vec_sents = [[word_id.get(w, 0) for w in sent] for sent in docMat]
    return vec_sents

def vectorize_sent(sent):
    """takes sentence array of words and return vectorize sentence array
       <UNK> token -> 0 for unknown words
    """
    vec_sent = [word_id.get(w, 0) for w in sent]
    return vec_sent


EMBEDDING_DIM = 150

embedding_matrix = np.zeros((len(word_id) + 1, EMBEDDING_DIM))
for word, i in word_id.items():
    #print word, i
    embedding_vector = word_vecs.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


def conv_encoder(nb_filters, filter_len, in_shape):
    """model to encode a sentence with a particular filter size
       nb_filters: number of filters, filter_len: size of filter,
       in_shape: input shape (_, SENT_LEN, WORD_LEN)
    """
    model = Sequential()
    model.add()
    model.add(MaxPooling1D(pool_length=2, stride=None, border_mode='same'))
    return model

def make_sentence_encoder(nb_filters, filter_lens, emb_size, in_shape):
    """model to encode a sentence with different filter sizes
       basically, learns different types of representations based
       on filter sizes like unigram, bigram, trigram etc.
       @in_shape: (_, SENT_LEN, EMB_LEN)
    """
    graph_in = Input(shape=in_shape)
    convs = []
    for flt in map(lambda x: x*emb_size, filter_lens):
        conv = Convolution1D(nb_filter=nb_filters, filter_length=flt, border_mode='same',
                             activation='relu')(graph_in) # subsample_length=1
        # pool one from each feature map
        #pool = MaxPooling1D(pool_length=in_shape[1]*emb_size-flt+1)(conv)
        pool = GlobalMaxPooling1D()(conv)
        #pool = MaxPooling1D(pool_length=32)(conv)
        #flatten = Flatten()(pool)
        convs.append(pool)

    if len(filter_lens)>1:
        out = Merge(mode='sum')(convs)
    else:
        out = convs[0]

    graph = Model(input=graph_in, output=out)
    return graph


sent_encoder = None
embedding_layer = None

def encode_all_sentences(vocab_size, dense_emb_dim, doc_maxlen, sent_maxlen, nb_filters, filter_lens, in_shape, out_dim):
    """A model that encodes each sentences using conv nets.
       Returns encoded set of sentences.
       Input: Document,  SHAPE: (_, DOC_LEN, SENT_LEN)
       Output: SHAPE: (_, DOC_LEN, NEW_SENT_LEN)
    """
    # setup a sentence encoder
    global sent_encoder, embedding_layer
    sent_encoder = make_sentence_encoder(nb_filters, filter_lens, dense_emb_dim, in_shape)
    print "+++++++++"
    print sent_encoder.input_shape
    print sent_encoder.output_shape
    print "+++++++++"

    # embed the input sequence into a sequence of vectors
    sentences_encoder = Sequential()
    # initialize embedding layer with wordvecs
    #sentences_encoder.add(Embedding(input_dim=vocab_size,
    #                              output_dim=dense_emb_dim,
    #                              input_length=sent_maxlen))
    embedding_layer = Embedding(input_dim=vocab_size, output_dim=dense_emb_dim,
                            weights=[embedding_matrix], trainable=False)
    sentences_encoder.add(TimeDistributed(embedding_layer,
                                          input_shape=(doc_maxlen, sent_maxlen),
                                          input_dtype='int32'))
    sentences_encoder.add(TimeDistributed(Dropout(0.3)))

    sentences_encoder.add(TimeDistributed(sent_encoder))
    sentences_encoder.add(TimeDistributed(Dense(out_dim)))
    return sentences_encoder

def make_doc_encoder(inseq_len, in_dim, out_seq_len):
    """model to encode a document. makes use of convlstm
       returns a vector of dim out_seq_len
    """
    # encoded sentences feed to lstm
    doc_encoder = Sequential()
    doc_encoder.add(LSTM(out_seq_len, input_shape=(inseq_len, in_dim)))
    return doc_encoder


################ Initialize variables ###################
vocab_size = len(word_id) + 1
dense_emb_dim = 150 # google word2vec embedding size
doc_len = 50
sent_len = 60
nb_filters = 128
filter_lens = [1, 2, 3, 4, 5]
in_shape = (sent_len, dense_emb_dim)
sent_out_dim = 128
doc_out_dim = 128


# setup input
doc_inpt = Input(shape=(doc_len, sent_len))

################ Document Encoding ######################
# encode all sentences
sentences_encoder = encode_all_sentences(vocab_size, dense_emb_dim, doc_len, sent_len, nb_filters,
                                          filter_lens, in_shape, sent_out_dim)

# get sentence encodings for document
out_enc = sentences_encoder(doc_inpt)

# make document encoding
doc_encoder = make_doc_encoder(doc_len, sent_out_dim, doc_out_dim)

doc_enc = doc_encoder(out_enc)          # stage 1 : Document encodings done


############### Sentence Encoding #####################
#sent_inpt = Input(shape=(sent_len, dense_emb_dim))
sent_input = Input(shape=(sent_len,), dtype='int32')
# embedding_layer = Embedding(input_dim=vocab_size, output_dim=dense_emb_dim,
#                             weights=[embedding_matrix], trainable=False)

embedded_sent = embedding_layer(sent_input)

# encode the sentence
sent_enc = sent_encoder(embedded_sent)           # stage 2 : Sentence encoding done


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

def compute_accuracy(predictions, labels):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return labels[predictions.ravel() < 0.5].mean()


############ Make base Network #################
def create_base_network(input_dim):
    '''Base network to be shared (eq. to feature extraction).
    '''
    seq = Sequential()
    seq.add(Dense(128, input_shape=(input_dim,), activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(128, activation='relu'))
    return seq


# network definition
input_dim = 128 # should be size of doc_enc/sent_enc
base_network = create_base_network(input_dim)

#input_a = Input(shape=(input_dim,))
#input_b = Input(shape=(input_dim,))
input_a = doc_enc
input_b = sent_enc

processed_a = base_network(input_a)
processed_b = base_network(input_b)

distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([processed_a, processed_b])

model = Model(input=[doc_inpt, sent_input], output=distance)


print model.summary()


# train
adam = Adam()
model.compile(loss=contrastive_loss, optimizer=adam)

from keras.utils import np_utils
from keras.preprocessing import sequence
from lxml import etree


# # temporary setting
# MAX_DOC_LEN = 100
# MAX_SENT_LEN = 150
# nb_class = 2

class DataServer(object):
    """ Facilitates data serving in batches
    """
    def __init__(self, config):
        self.MAX_DOC_LEN = config['MAX_DOC_LEN']
        self.MAX_SENT_LEN = config["MAX_SENT_LEN"]
        self.nb_class = config["nb_class"]

    def serve_batchData(self, filepath, serve_size):
        """@serve_size: number of items to process as batch
           returns X and Y
        """
        X_doc, X_sent, Y = [], [], []
        # check = []
        it = 1

        doc_data = ""
        sent_data = ""
        labels = []

        for event, elem in etree.iterparse(filepath, events=('start', 'end', 'start-ns', 'end-ns')):
            if elem.tag == "DOC" and event == "end":
                doc_data = elem.text.strip()
                name = elem.attrib["name"]

            elif elem.tag == "SENT" and event == "end":
                sent_data = elem.text.strip()

            elif elem.tag == "LABEL" and event == "end":
                label = int(elem.text)

                # got all data
                # Vectorize these data
                sent = [w for w in sent_data.split()]
                vsent = vectorize_sent(sent)

                doc = [[w for w in s.split()] for s in doc_data.split("\n")]
                vdoc = vectorize_document(doc)

    #             print sent
    #             print doc
    #             print vsent
    #             print vdoc
    #             print "###################"
    #             print label
    #             print "+++++++++DOC Data+++++++++++++"
    #             print doc_data
    #             print doc_data.split("\n")

                #break

                # It's safe to call clear() here because no descendants will be
                # accessed
                elem.clear()
                # Also eliminate now-empty references from the root node to elem
                for ancestor in elem.xpath('ancestor-or-self::*'):
                    while ancestor.getprevious() is not None:
                        del ancestor.getparent()[0]


                ########## Padding document matrix and sentence array ##########
                # print "Initial length of vdoc", len(vdoc)
                diff = self.MAX_DOC_LEN - len(vdoc)

                if diff > 0:
                    # padding
                    for i in xrange(diff):
                        # labe
                        vdoc.append([0.]* self.MAX_SENT_LEN)
                elif diff < 0:
                    # truncate
                    vdoc = vdoc[:diff]

    #             print vdoc
    #             print "++++++++++++++++++++++++++++++++++++"
    #             print "final length of vdoc", len(vdoc)
                new_vdoc = sequence.pad_sequences(vdoc, maxlen=self.MAX_SENT_LEN, padding='post', truncating='post')
    #             print new_vdoc
    #             print "length of sentence", len(new_vdoc[0])


                # padding sentence array
                #print "******************************************"
                lim = self.MAX_SENT_LEN - len(vsent)
                #print "previous sent len", len(vsent)
                if lim > 0:
                    # padd
                    for i in xrange(lim):
                        vsent.append(0.)
                elif lim < 0:
                    # truncate
                    vsent = vsent[:lim]

                #print vsent
                #print "new sent array len", len(vsent)
                #print new_vdoc[0]

                # check
    #             if np.all(vsent == new_vdoc[0]):
    #                 print "OK"

                X_doc.append(new_vdoc)
                X_sent.append(vsent)
                Y.append(label)

                if (it % serve_size) == 0:
                    Y = np_utils.to_categorical(Y, self.nb_class)
    #                 print len(Y)
    #                 print len(Y[0])
    #                 print Y[0]
                    X_doc, X_sent, Y = np.array(X_doc), np.array(X_sent), np.array(Y)
                    yield X_doc, X_sent, Y
                    X_doc, X_sent, Y = [], [], []
                    #break
                it += 1
        # data which are left
        if X_doc and X_sent and Y:
            Y = np_utils.to_categorical(Y, nb_class)
            X_doc, X_sent, Y = np.array(X_doc), np.array(X_sent), np.array(Y)
            yield X_doc, X_sent, Y


ftrain = 'doc-sent-data-training.xml'
fval = 'doc-sent-data-validation.xml'

nb_epochs = 20

config = {"MAX_DOC_LEN": 50,
          "MAX_SENT_LEN": 60,
          "nb_class": 2}

ds_train = DataServer(config)
ds_val = DataServer(config)

# make training data server
#tbatch_data = ds_train.serve_batchData(ftrain, 35)

# make validation dataserver; take full data
#val_data = ds_val.serve_batchData(fval, 10)

# X1, X2, Y = tbatch_data.next()
# print X1.shape
# print X2.shape
# print Y.shape

################## Test ##################
# make training data server
#tbatch_data = ds_train.serve_batchData(ftrain, 35)
#make validation dataserver; take full data
#val_data = ds_val.serve_batchData(fval, 10)

#X1, X2, Y = tbatch_data.next()
#print X1.shape
#print X2.shape
#print Y.shape

# loading weights to the model
print "loading weights..."
model.load_weights("weights_main_0_0.25_12000")
print "finished loading weights."
tflag = 1

for epoch in xrange(nb_epochs):
    t_data = ds_train.serve_batchData(ftrain, 32)
    v_data = ds_val.serve_batchData(fval, 10000)
    VX_d, VX_s, V_Y = v_data.next()
    count = 0

    if tflag:
	count = 12000
        tflag = 0
    print "In epoch %s" %(epoch)
    for X_d, X_s, Y in t_data:
	#print "starting batch training..."
        loss = model.train_on_batch([X_d, X_s], Y)
        count += 1
        print "epoch %s training loss: %s" %(epoch, loss)
        # after 1000 batches print info
        if ((count % 100) == 0):
            print "epoch %s training loss after 100 batches: %s" %(epoch, loss)

	if ((count % 1000) == 0):
	    print "saving weights after %s batch" %(count)
	    model.save_weights('weights_main_'+str(epoch)+"_"+str(loss)+"_"+str(count))

	# use only batches as training data
	if ((count % 15000) == 0):
	    break


    # compute accuracy after each epoch on training and validation sets
    #pred = model.predict([X_d, X_s])
    #tr_acc = compute_accuracy(pred, tr_y)
    print "predicting..."
    pred = model.predict([VX_d, VX_s])
    #print "after predicting..."
    val_acc = compute_accuracy(pred, V_Y)
    #print "after val_acc...."

    #print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
    print('* Accuracy on validation set: %0.2f%%' % (100 * val_acc))

    print "saving weigths after epoch %s" %(epoch)
    model.save_weights('out_weights_main_'+str(epoch)+"_"+str(loss)+"_"+str(val_acc))


# model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y,
#           validation_data=([te_pairs[:, 0], te_pairs[:, 1]], te_y),
#           batch_size=128,
#           nb_epoch=nb_epoch)

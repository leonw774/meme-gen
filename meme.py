import os
import random
import numpy as np

from imageio import imread
from gensim.models import word2vec

from keras import optimizers, applications
from keras import backend as K
from keras.models import Model
from keras.layers import Activation, Concatenate, Dense, Dropout, Flatten, Input, Lambda, LSTM, merge, multiply, Permute, RepeatVector, TimeDistributed

from config import *
CREATE_NEW_W2V = True

train_image_file_list = [("train/images/" + i) for i in os.listdir("train/images/")]
train_caption_file_list = [("train/p_captions/" + i) for i in os.listdir("train/p_captions/")]
test_image_file_list = [("test/images/" + i) for i in os.listdir("test/images/")]

#########################################
# Prepare Training Data
#########################################
total_word_count = 0
caption_list = []
joke_list = []
print("\npreparing sentence to train...")

# captions' sentence
for filename in train_caption_file_list :
    try :
        line_list = open(filename, 'r', encoding = 'utf-8-sig').readlines()
    except :
        print("No processed caption found. Please run process_captions.py first.")
        exit()
    caption = []
    for line in line_list :
        word_list = line.lower().split(" ") # 對每一行，先變小寫，再用空格分成list
        total_word_count += len(word_list)
        caption.extend(word_list) # 用extend來push是因為要push list-of-string into a list-of-string
    caption.append(ENDING_MARK) # 用append來push是因為要push string into list-of-string
    if len(caption) > MAX_LENGTH : MAX_LENGTH = len(caption)
    caption_list.append(caption) # 用append來push是因為要push list-of-string into a "list of list-of-string"
#print(caption_list[random.randint(0, 100)])

#########################################
# Train W2V (if True)
#########################################
if CREATE_NEW_W2V :
    w2v_train_sentence_list = []
    w2v_train_sentence_list.extend(caption_list)
    print("training w2v model...")
    word_model = word2vec.Word2Vec(w2v_train_sentence_list,
        iter = 24,
        sg = 1,
        size = WORD_VEC_SIZE,
        window = 5,
        workers = 4,
        min_count = 1)
    word_model.save("meme_word2vec.model")
else :
     word_model = word2vec.Word2Vec.load("meme_word2vec.model")
del w2v_train_sentence_list
word_vector = word_model.wv
VOCAB_SIZE = word_vector.syn0.shape[0]
print("vector size: ", WORD_VEC_SIZE)
print("vocab size: ", VOCAB_SIZE)
print("total_word_count:", total_word_count)
#print(word_vector.most_similar("我", topn = 10))

def make_sentence_matrix(word_list) :
    input_matrix  = np.zeros([1, len(word_list) + 1, WORD_VEC_SIZE], dtype=np.int32)
    target_matrix = np.zeros([1, len(word_list) + 1, 1], dtype=np.int32)
    i = 0
    for word in word_list :
        try :
            # input_matrix begin at 1 because index 0 is zero vector as starting symbol
            input_matrix[0, i + 1] = word_vector[word]
            target_matrix[0, i, 0] = word_vector.vocab[word].index # because sparse_categorical
        except KeyError :
            continue
        i += 1
    return input_matrix, target_matrix

def get_image(img_filename, resize = None) :
    #print(img_filename)
    img = imread(img_filename)
    if (len(img.shape) == 2) : # for gray scale
        img = np.expand_dims(img, axis = 2)
        img = np.broadcast_to(img, (img.shape[0], img.shape[1], 3))
    elif img.shape[2] == 4: # for PNG, GIF
        img = img[:, :, 0 : 3]
    img = np.expand_dims(img, axis = 0) / 255.0
    if resize : img = img.reshape(resize)
    return img

#########################################
# Paired Training Data Generator
#########################################
def gen_pair_training_data(resize = None) :
    while(True) :
        for i, img_filename in enumerate(train_image_file_list) :
            #print(i, img_filename)
            img = get_image(img_filename)
            input_cap, target_cap = make_sentence_matrix(caption_list[i])
            yield [img, input_cap], target_cap
            # end if count
        # end for train_image_file_list
    #end infinite while
# end def

#########################################
# Build Model
#########################################
### Image Encoder ###
Imagenet = applications.inception_v3.InceptionV3()
#Imagenet = applications.mobilenet.MobileNet()
Imagenet.trainable = False
# 使用 MobileNet 是因為它佔用空間最小(17MB)
# 論文裡，同Google的im2txt模型，用的是 InceptionV3

image_in = Input([None, None, 3])
classes = Imagenet(image_in)
state_in = Dense(LSTM_UNIT)(classes)

### LSTM Decoder ###
cap_in = Input([None, WORD_VEC_SIZE])
zeros = Lambda(lambda x: K.zeros_like(x), output_shape = lambda s: s)(state_in)
# 上面這個Lambda的output是與input形狀相同的零矩陣
# LSTM initial state: [hidden_state, memory_cell_state]; default is zero vectors
lstm_out = LSTM(LSTM_UNIT, return_sequences = True, stateful = False) (cap_in, initial_state = [state_in, zeros])
print(lstm_out.shape) # (BATCH, TIME_STEP, LSTM_UNIT)
cap_out = Dense(VOCAB_SIZE, activation = "softmax")(lstm_out)

# Optimizers
# 論文裡說試了SGD和Mometum，然後SGD不錯
sgd = optimizers.sgd(lr = 0.02)
sgd_nesterov = optimizers.sgd(lr = 0.1, momentum = 0.9, nesterov = True)
adam = optimizers.Adam(lr = 0.01)

# Metrics: Perplexity
# 反正好像就是自然常數的熵次方啦，懶得看數學推導
def sparse_categorical_perplexity(y_true, y_pred) :
    return K.exp(K.sparse_categorical_crossentropy(y_true, y_pred))

# Define Model
MemeGen = Model([image_in, cap_in], cap_out)
MemeGen.summary()
MemeGen.compile(loss = "sparse_categorical_crossentropy",
                metrics = [sparse_categorical_perplexity],
                optimizer = sgd_nesterov)

#########################################
# Train Model
#########################################
# keras不給我用train_on_batch，因為每張圖大小不一樣，算了沒差
image_data_size = len(train_image_file_list)
meme_hisory = MemeGen.fit_generator(generator = gen_pair_training_data(),
                      steps_per_epoch = image_data_size,
                      epochs = EPOCHS,
                      verbose = 1)

#########################################
# Predict Test
#########################################
def sample(prediction, temperature = 1.0) :
    prediction = np.asarray(prediction).astype('float64')
    prediction = np.log(prediction) / temperature
    exp_preds = np.exp(prediction)
    prediction = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, prediction, 1)
    return probas

for test_img_name in test_image_file_list :
    print(test_img_name)
    pred_sentence = ""
    for i in range(MAX_LENGTH) :
        input_matrix, _ = make_sentence_matrix(pred_sentence)
        pred = MemeGen.predict([get_image(test_img_name), input_matrix])
        pred = sample(pred[0, -1], temperature = 0.7)
        pred_word = word_vector.wv.index2word[np.argmax(pred[0])]
        if pred_word == ENDING_MARK : break
        pred_sentence += pred_word
        if pred_word == ENDING_MARK : continue
    open("test/captions/" + test_img_name[12:-3] + "txt", "w+",  encoding = 'utf-8-sig').write(pred_sentence)

#print(meme_hisory.history)

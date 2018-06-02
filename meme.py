import os
import random
import numpy as np

from imageio import imread
from gensim.models import word2vec
from keras import optimizers
from keras import backend as K
from keras import applications
from keras.models import Model
from keras.layers import Activation, Concatenate, Dense, Dropout, Embedding, Flatten, Input, Lambda, LSTM

CREATE_NEW_W2V = True
WORD_VEC_SIZE = 200
LSTM_DIM = 128
EPOCHS = 10
ENABLE_RANDOM_TRAINING = False


training_image_file_list = [("images/" + i) for i in os.listdir("images/")]
training_caption_file_list = [("p_captions/" + i) for i in os.listdir("captions/")]
training_jokes_file_list = [("p_jokes/" + i) for i in os.listdir("p_jokes/")]
testing_image_file_list = [("test/images/" + i) for i in os.listdir("test/images/")]

#########################################
# Prepare W2V
#########################################
total_word_count = 0
if CREATE_NEW_W2V :
    w2v_train_sentence_list = []
    print("\npreparing sentence to train...")
    # captions' sentence
    for filename in training_caption_file_list :
        line = open(filename, 'r', encoding = 'utf-8-sig').read().split(" ")
        total_word_count += len(line)
        w2v_train_sentence_list.append(line)
    # jokes' sentence
    for filename in training_jokes_file_list :
        line_list = open(filename, 'r', encoding = 'utf-8-sig').readlines()
        post_sentence = []
        for line in line_list :
            line = line.split(" ")
            if len(line) <= 1 : continue
            total_word_count += len(line)
            post_sentence.extend(line)
        w2v_train_sentence_list.append(post_sentence)
    print("training w2v model...")
    word_model = word2vec.Word2Vec(w2v_train_sentence_list,
        iter = 12,
        sg = 1,
        size = WORD_VEC_SIZE,
        window = 5,
        workers = 4,
        min_count = 3)
    word_model.save("meme_word2vec_by_char.model")
    del w2v_train_sentence_list
else :
    word_model = word2vec.Word2Vec.load("meme_word2vec_by_char.model")
word_vector = word_model.wv
VOCAB_SIZE = word_vector.syn0.shape[0]
print("\nvector size: ", WORD_VEC_SIZE, "\nvocab size: ", VOCAB_SIZE, "\ntotal_word_count:", total_word_count)
print(word_vector.most_similar("怕", topn = 10))

#########################################
# Trainging Data Preparation
#########################################
caption_list = []
for fname in training_caption_file_list :
    line_list = open(fname, 'r', encoding = 'utf-8-sig').readlines()
    caption = []
    for line in line_list :
        word_list = line.split() + ['\n']
        total_word_count += len(word_list)
        if len(word_list) <= 1 : continue
        caption += word_list
    caption_list.append(caption)

joke_list = []
for fname in training_jokes_file_list :
    line_list = open(fname, 'r', encoding = 'utf-8-sig').readlines()
    post_sentence = []
    for line in line_list :
        word_list = line.split() + ['\n']
        total_word_count += len(word_list)
        if len(word_list) <= 1 : continue
        post_sentence += word_list
    joke_list.append(post_sentence)

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
    img = imread(img_filename)
    if (len(img.shape) == 2) :
        img = np.expand_dims(img, axis = 2)
        img = np.broadcast_to(img, (img.shape[0], img.shape[1], 3))
    elif img.shape[2] == 4:
        img = img[:, :, 0 : 3]
    img = np.expand_dims(img, axis = 0) / 255.0
    #print(img_filename)
    if resize : img = img.reshape(resize)
    return img

#########################################
# Paired Training Image Data Generator
#########################################
def generator_pair_training_data(resize = None) :
    while(True) :
        for i, img_filename in enumerate(training_image_file_list) :
            img = get_image(img_filename)
            input_cap, target_cap = make_sentence_matrix(caption_list[i])
            yield [img, input_cap], target_cap

#########################################
# Random Training Image Data Generator
# (random image with random jokes)
#########################################
def generator_random_training_data(resize = None) :
    while(True) :
        img_filename = random.choice(training_image_file_list)
        img = get_image(img_filename)
        joke_line = random.choice(joke_list)
        input_cap, target_cap = make_sentence_matrix(joke_line)
        yield [img, input_cap], target_cap


#########################################
# Build Model
#########################################
# Image Encoder
Imagenet = applications.mobilenet.MobileNet()
Imagenet.trainable = False
# 使用 MobileNet 是因為它佔用空間最小(17MB)
# 論文裡，同Google的im2txt模型，用的是 InceptionV3
# https://github.com/tensorflow/models/tree/master/research/im2txt/

image_in = Input([None, None, 3])
classes = Imagenet(image_in)
state_in = Dense(LSTM_DIM)(classes)

# LSTM Decoder
caption_in = Input([None, WORD_VEC_SIZE])
zeros = Lambda(lambda x: K.zeros_like(x), output_shape = lambda s: s)(state_in)
# lstm initial state: [hidden_state, memory_cell_state]; default is zero vectors
x = LSTM(LSTM_DIM, return_sequences = True, stateful = False)(caption_in, initial_state = [state_in, zeros])
x = Dropout(0.2)(x)
caption_out = Dense(VOCAB_SIZE, activation = "softmax")(x)

# Model
MemeGen = Model([image_in, caption_in], caption_out)
MemeGen.summary()
sgd_nesterov = optimizers.sgd(lr = 0.01, momentum = 0.9, nesterov = True)
adam_05 = optimizers.Adam(lr = 0.001, beta_1 = 0.5)
MemeGen.compile(loss = 'sparse_categorical_crossentropy', optimizer = adam_05)

#########################################
# Train Model
#########################################
image_data_size = len(training_image_file_list)
joke_data_size = len(training_jokes_file_list) // EPOCHS
for epoch in range(EPOCHS) :
    print(epoch, "/", EPOCHS)
    if ENABLE_RANDOM_TRAINING :
        MemeGen.fit_generator(generator = generator_random_training_data(),
            steps_per_epoch = joke_data_size,
            epochs = 1,
            verbose = 2)
    MemeGen.fit_generator(generator = generator_pair_training_data(),
        steps_per_epoch = image_data_size,
        epochs = 1,
        verbose = 2)

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

for test_img_name in testing_image_file_list :
    print(test_img_name)
    pred_sentence = ""
    for i in range(32) :
        input_matrix, _ = make_sentence_matrix(pred_sentence)
        pred = MemeGen.predict([get_image(test_img_name), input_matrix])
        pred = sample(pred[0, -1], temperature = 0.7)
        pred_word = word_vector.wv.index2word[np.argmax(pred[0])]
        pred_sentence += pred_word
        #if pred_word == "\n" and pred_sentence != "" : continue
    open("test/captions/" + test_img_name[12:-3] + "txt", "w+",  encoding = 'utf-8-sig').write(pred_sentence)

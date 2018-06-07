from keras.layers import Dense, Dropout, Input, Flatten, Reshape
from keras.layers import Conv2D, MaxPool2D, Embedding
from keras.models import Model, model_from_yaml
import dataHelper
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pickle
import pandas as pd

train_filename = "C:/Users/dkaraca2/Desktop/SentenceSimilarity/suggestion_cnn/laptop_training_set.csv"
path_to_w2v = "C:/Users/dkaraca2/Desktop/SentenceSimilarity/suggestion_cnn/w2v_laptop_reviews.pickle"
dim_emb = 100
filter_size = 3
num_filters = 50
drop = 0.5

def create_model(seq_len,
				 vocab_size,dim_emb,
				 weights=None,
				 filter_size=3,num_filters=50,
				 drop=0.5):
	inputs = Input(shape=(seq_len,), dtype='int32')

	embedding = Embedding(input_dim=vocab_size+1, output_dim=dim_emb, input_length=seq_len, weights=[weight_matrix], trainable=False)(inputs)

	reshape = Reshape((seq_len,dim_emb,1,))(embedding)

	conv = Conv2D(num_filters, kernel_size=(filter_size,dim_emb), padding='valid', kernel_initializer='uniform', activation='relu', data_format='channels_last')(reshape)

	pool = MaxPool2D(pool_size=(seq_len-filter_size+1,1), strides=(1,1))(conv)

	flatten = Flatten()(pool)

	dropout = Dropout(drop)(flatten)

	output = Dense(2, activation='softmax')(dropout)

	model = Model(inputs=inputs,outputs=output)

	model.compile(optimizer='adam', loss='mean_squared_error', metrics=['binary_accuracy'])

	return model

with open(path_to_w2v,'rb') as f:
	w2vModel = pickle.load(f)

text_raw, labels, max_len_train = dataHelper.load_text_and_labels(train_filename)
train_seqs, tokenizer = dataHelper.create_vocab_and_word_sequences(text_raw,max_len_train)
w2v_embMatrix = dataHelper.create_word2vec_matrix(tokenizer,w2vModel)
vocab_size = len(list(tokenizer.word_index.keys()))

print("DATA LOADED...")

cnn = KerasClassifier(build_fn=create_model,seq_len=max_len_train,
					  vocab_size=vocab_size,dim_emb=dim_emb,
					  weights=w2v_embMatrix,
					  filter_size=filter_size,num_filters=num_filters,
					  drop=drop,
					  epochs=5,batch_size=32)


## THIS PART NEEDS TO CHANGE
print("STARTING TRAINING...")
print (type(train_seqs))
stratified_kf = StratifiedKFold(n_splits=10)
stratified_kf.get_n_splits(train_seqs,labels)
scores = cross_val_score(cnn, list(train_seqs), labels, cv=stratified_kf, verbose=True)

print("STRATIFIED KF SPLITS...")
print(stratified_kf)
print("CV SCORES...")
print(scores)

model_yaml = model.model.to_yaml()
with open('modelCNN_05242018.yaml','w') as yaml_file:
	yaml_file.write(model_yaml)
model.model.save_weights('modelCNN_05242018_weights.h5')
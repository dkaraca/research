from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, Input
import dataHelper

def create_model(weights=None):
	inputs = Input(shape=(None,100,),dtype='float32')
	embedding = Embedding(weights=[weight_matrix])
	#mask = Masking(mask_value=1500,input_shape=(seq_len,dim_emb,))(inputs)
	lstm = LSTM(100,activation='tanh',recurrent_activation='softmax')(inputs)
	#dropout = Dropout(drop)(lstm)
	output = Dense(2,activation='softmax')(lstm)
	model = Model(inputs=inputs,outputs=output)
	model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mse'])
	print(model.summary())
	return model



## CURRENTLY WORKING ON THIS MODEL APPROACH
text_raw, labels, max_len_train = dataHelper.load_text_and_labels(train_filename)
train_seqs, tokenizer = dataHelper.create_vocab_and_word_sequences(text_raw,max_len_train)
w2v_embMatrix = dataHelper.create_word2vec_matrix(tokenizer,w2vModel)
vocab_size = len(list(tokenizer.word_index.keys()))
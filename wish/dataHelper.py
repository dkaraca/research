import pandas as pd
from nltk import word_tokenize, pos_tag
import numpy as np
import pickle
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences

def create_vocab_and_word_sequences(rawText,maxLen):
	#filters = [elem.encode('ascii') for elem in filters]
	#word_sequences = [text_to_word_sequence(text) for text in rawText]
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(rawText)
	seq_matrix = tokenizer.texts_to_sequences(rawText)
	seq_matrix = pad_sequences(seq_matrix,maxlen=maxLen,padding='post')
	print(seq_matrix)
	#print(list(seq_matrix[0]))
	#sequences = get_word_sequences(word_sequences,tokenizer,maxLen)
	return seq_matrix, tokenizer

def get_word_sequences(rawText,tokenizer,maxLen): # inputs are list of lowercase word sequences, tokenizer and maxlen
	output = tokenizer.texts_to_sequences(rawText)
	output = pad_sequences(output,maxlen=maxLen,padding='post',truncating='post')
	# output = []
	# for seq in wordSeqs:
	# 	curr = []
	# 	for elem in seq:
	# 		if elem in list(tokenizer.word_index.keys()):
	# 			curr.append(tokenizer.word_index[elem])
	# 		else:
	# 			curr.append(0)
	# 	output.append(curr)
	# 	#print(curr)
	# output = pad_sequences(np.asarray(output,dtype=np.int32),maxlen=maxLen,padding='post',truncating='post')
	return output

def create_word2vec_matrix(tokenizer,w2vModel):
	output = [None] * (len(list(tokenizer.word_index.keys()))+1)
	count=0
	output[0] = np.random.uniform(-0.05,0.05,size=(300,))
	#print(tokenizer.word_index.keys())
	for key in tokenizer.word_index.keys():
		try:
			output[tokenizer.word_index[key]] = w2vModel[key]
		except:
			count+=1
			output[tokenizer.word_index[key]] = np.random.uniform(-0.05,0.05,size=(300,))
	print(np.array(output).shape)
	print("HERE")
	print(count)
	return np.array(output)

def load_text_and_labels(filename):
	print("STARTING TO LOAD TRAINING SET")
	max_len = 0
	df = pd.read_csv(filename)
	#text = [text_to_word_sequence(sentence, filters=False, lower=True, split=' ') for sentence in df['Sentence'].tolist()]
	text = [word_tokenize(sentence.lower()) for sentence in df['Sentence'].tolist()]
	# text = []
	text_raw = df['Sentence'].tolist()
	# for sentence in df['Sentence'].tolist():
	# 	filtered_words = [word for word in word_tokenize(sentence.lower()) if word not in stopwords]
	# 	text.append(filtered_words)
	# 	text_raw.append(' '.join(elem for elem in filtered_words))
	#print(text_raw)
	labels = []
	try:
		for label in df['Class'].tolist():
			if label == 0:
				labels.append(0)
			elif label == 1:
				labels.append(1)
	except:
		labels = []
	print("TRAINING SET LOADED")
	print(len(text),len(labels))
	print(labels[:100])
	return text_raw, labels, max([len(arr) for arr in text])

def load_text_and_labels_with_pos(filename):
	print("STARTING TO LOAD TRAINING SET")
	max_len = 0
	df = pd.read_csv(filename)
	#text = [text_to_word_sequence(sentence, filters=False, lower=True, split=' ') for sentence in df['Sentence'].tolist()]
	text = [pos_tag(word_tokenize(sentence.lower())) for sentence in df['Sentence'].tolist()]
	labels = []
	try:
		for label in df['Class'].tolist():
			if label == 0:
				labels.append([1,0])
			else:
				labels.append([0,1])
	except:
		labels = []
	print("TRAINING SET LOADED")
	print(len(text),len(labels))
	return text, np.array(labels), max([len(arr) for arr in text])


def padArray(arr,seq_len):
	result = np.zeros((seq_len,arr.shape[1]))
	result[:arr.shape[0],:arr.shape[1]] = arr
	return result


def generate_emb_sequences(word_list,model,dim,seq_len):
	count = 0
	print("STARTING EMBEDDING LOOKUP")
	emb_seq = []
	for i in range(len(word_list)):
		curr_seq = []
		#print("DONE")
		for word in word_list[i]:
			# if word.lower() in model.vocab:
			try:
				curr_seq.append(model[word.lower()])
			except:
				count+=1
				continue
			# else:
			# 	random_arr = np.random.uniform(-0.05,0.05,(dim,))
			# 	curr_seq.append(random_arr)
		emb_seq.append(padArray(np.array(curr_seq).reshape(len(curr_seq),dim),seq_len))
		if len(emb_seq) % 1000 == 0:
			print(len(emb_seq))
	print(count)
	emb_seq = np.float32(np.array(emb_seq)).reshape((len(word_list),seq_len,dim))
	return emb_seq

def generate_emb_sequences_with_pos(word_list,model_words,model_pos,dim,seq_len):
	print("STARTING EMBEDDING LOOKUP")
	emb_seq = []
	print(len(word_list))
	for i in range(len(word_list)):
		curr_seq = []
		curr_pos_seq = []
		curr_seq = []
		#print("DONE")
		for word in word_list[i]:
			if word[0].lower() in model_words.vocab:
				curr_seq.append(model_words[word[0].lower()])
			else:
				random_arr = np.random.uniform(-0.05,0.05,(int(dim/2),))
				curr_seq.append(random_arr)
			if word[1] in model_pos.vocab:
				curr_pos_seq.append(model_pos[word[1]])
			else:
				curr_pos_seq.append(np.random.uniform(-0.05,0.05,(int(dim/2),)))
			
			conc_seq = np.concatenate((curr_seq,curr_pos_seq),axis=1)

		emb_seq.append(padArray(np.array(conc_seq).reshape(conc_seq.shape[0],dim),seq_len))
			
		if len(emb_seq) % 1000 == 0:
			print(len(emb_seq))
	emb_seq = np.float32(np.array(emb_seq)).reshape((len(word_list),seq_len,dim))
	return emb_seq
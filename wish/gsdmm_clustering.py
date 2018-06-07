from nltk.stem import WordNetLemmatizer
import pandas as pd
from gsdmm import MovieGroupProcess
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv('laptop_training_set.csv')
sentences = df['Sentence'].tolist()
labels = df['Class'].tolist()

lem = WordNetLemmatizer()
vect = CountVectorizer(min_df=5,max_df=0.95)

with open('C:/Users/dkaraca2/Desktop/SentenceSimilarity/stopwords.txt','r') as f:
	content = f.readlines()
	stopwords = [c.strip() for c in content]

samples = []
samples_raw = []
left_out = []
labels_samples = []
labels_left_out = []

lemmatized = []
for sentence in sentences:
	curr = [lem.lemmatize(token) for token in sentence.split() if token not in stopwords]
	lemmatized.append(' '.join(elem for elem in curr))

vect.fit_transform(lemmatized)

for i in range(len(sentences)):
	curr = [token for token in sentences[i].split() if token in vect.vocabulary_]
	if len(curr) > 0:
		samples.append(curr)
		samples_raw.append(sentences[i])
		labels_samples.append(labels[i])
	else:
		left_out.append(sentences[i])
		labels_left_out.append(labels[i])
print(len(vect.vocabulary_))

mgp = MovieGroupProcess(K=40, alpha=0.1, beta=0.1, n_iters=30)
mgp.fit(samples,len(vect.vocabulary_))

results = [mgp.choose_best_label(sample) for sample in samples]
if len(results) < len(sentences):
	results.extend(["Left out"] * (len(sentences)-len(results)))

pd.DataFrame({'Sentence':samples_raw+left_out, 'Label':labels_samples+labels_left_out, 'Cluster':results}).to_csv('clustering_results.csv')

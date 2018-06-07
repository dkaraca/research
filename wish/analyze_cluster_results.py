import pandas as pd
from collections import Counter

df = pd.read_csv('clustering_results.csv')

with open('C:/Users/dkaraca2/Desktop/SentenceSimilarity/stopwords.txt','r') as f:
	content = f.readlines()
	stopwords = [c.strip() for c in content]

byCluster = df.groupby('Cluster')

numPositive = {}
numNegative = {}
topWords = {}

for group in byCluster.groups:
	curr = byCluster.get_group(group)
	numPositive[int(group)] = curr['Label'].tolist().count(1)
	numNegative[int(group)] = curr['Label'].tolist().count(0)
	all_words = []
	for sentence in curr['Sentence'].tolist():
		words = [token for token in sentence.split() if token not in stopwords]
		all_words.extend(words)
	topWords[int(group)] = Counter(all_words).most_common(15)

pd.DataFrame({'Cluster':topWords.keys(),'Top Words':topWords.values(),'Positive':numPositive.values(),'Negative':numNegative.values()}).to_csv('cluster_analysis.csv')
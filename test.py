import pickle
from numpy import *
from sklearn import tree
from sklearn import svm
from sklearn import linear_model
import nltk
from feature_extractor_5 import *
from post_process import *
if __name__ == '__main__':    
	test_c_feature_set=[]
	test_r_feature_set=[]
	#test_entailment=[];
	#test_rscore=[];
	test_pair_IDs=[]
	#load classifiers
	print "Loading classifiers..."
	clf_entailment=pickle.load(open("clf_entailment.p","rb"))
	clf_rscore=pickle.load(open("clf_rscore.p","rb"))

	print "Extracting features from test data..."
	fo=open("results.txt",'w')
	with open('SICK_trial.txt','r') as f_train:
		header=0;
		for line in f_train:
			line=line[:-1];
			if header!=0:
				[pair_ID,sentence_A,sentence_B,relatedness_score,entailment_judgement]=line.split('\t');
				r_feature_row=compute_r_features(sentence_A,sentence_B);
				c_feature_row=compute_c_features(sentence_A,sentence_B)
				#test_rscore.append(float(relatedness_score));
				#test_entailment.append(entailment_judgement);
				test_pair_IDs.append(pair_ID)
				test_r_feature_set.append(r_feature_row)			
				test_c_feature_set.append(c_feature_row)
			else:
				header=1

	print "Printing the results in results.txt"
	predicted_rscores=clf_rscore.predict(test_r_feature_set)
	predicted_entailments=clf_entailment.predict(test_c_feature_set)
	fo.write("pair_ID\trelatedness_score\tentailment_judgment\n")
	for i in xrange(len(test_pair_IDs)):
		fo.write(test_pair_IDs[i]+"\t"+str(final_score(predicted_rscores[i]))+"\t"+str(predicted_entailments[i])+"\n")
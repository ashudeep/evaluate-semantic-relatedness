from numpy import *
from sklearn import tree
from sklearn import svm
from sklearn import linear_model
from feature_extractor_5 import *
import nltk

if __name__ == '__main__':    
	r_feature_set=[]
	c_feature_set=[]
	label_entailment=[];
	label_rscore=[];
	#i=0
	with open('SICK_train.txt','r') as f_train:
		header=0;
		for line in f_train:
			line=line[:-1];
			#i+=1
			#print i
			if header!=0:
				[pair_ID,sentence_A,sentence_B,relatedness_score,entailment_judgement]=line.split('\t');
				r_feature_row=compute_r_features(sentence_A,sentence_B);
				c_feature_row=compute_c_features(sentence_A, sentence_B)
				label_rscore.append(float(relatedness_score));
				label_entailment.append(entailment_judgement);
				r_feature_set.append(r_feature_row);
				c_feature_set.append(c_feature_row)
			else:
				header=1;
			
	#train here on feature_set, label_entailment, label_rscore
	print "Features extracted. Now Training..."
	clf_entailment=svm.SVC(kernel='linear')
	clf_entailment.fit(c_feature_set,label_entailment);
	
	clf_rscore=svm.SVR(kernel='linear');
	clf_rscore.fit(r_feature_set,label_rscore);
	print "Saving the classifiers...."
	import pickle
	pickle.dump(clf_entailment,open("clf_entailment.p","wb"))
	pickle.dump(clf_rscore,open("clf_rscore.p","wb"))
	print "Training done."
	#print clf_entailment.predict(feature_row)
	#print clf_rscore.predict(feature_row)
	

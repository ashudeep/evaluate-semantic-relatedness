import nltk
from takelab_simple_features import *
from munkres import Munkres, print_matrix, make_cost_matrix
	
neg_words=["not","no","isn't","wasn't","don't","aren't","didn't","never","can't","without","few","less"];
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer();

from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
brown_ic = wordnet_ic.ic('ic-brown.dat')
#semcor_ic = wordnet_ic.ic('ic-semcor.dat')
pos_dict={'VB':wn.VERB,'VBD':wn.VERB,'VBG':wn.VERB,'VBN':wn.VERB,'VBP':wn.VERB,'VBZ':wn.VERB,'NN':wn.NOUN,'NNS':wn.NOUN,'NNP':wn.NOUN,'NNPS':wn.NOUN, 'JJ':wn.ADJ, 'JJR':wn.ADJ,'JJS':wn.ADJ,'RB':wn.ADV,'RBR':wn.ADV,'RBS':wn.ADV,'WRB':wn.ADV}
pos_list=['VB','VBG','VBD','VBN','VBP','VBZ','RB','RBR','RBS','JJ','JJR','JJS']
pos_dict2={'VB':'VERB','VBD':'VERB','VBG':'VERB','VBN':'VERB','VBP':'VERB','VBZ':'VERB','NN':'NOUN','NNS':'NOUN','NNP':'NOUN','NNPS':'NOUN', 'JJ':'ADJ', 'JJR':'ADJ','JJS':'ADJ','RB':'ADV','RBR':'ADV','RBS':'ADV','WRB':'ADV'}


def avg(a,b):
	return (a+b)/2
def clean(word):
    return word.strip(",. ")

def bleu(text, entailment):
    def ngrams(stringlist, n):
        i = 0
        while i + n <= len(stringlist):
            yield tuple(stringlist[i:i+n])
            i += 1
    words = [clean(x) for x in text.lower().split()]
    hwords = [clean(x) for x in entailment.lower().split()]
    bleus = 0.0
    for N in range(1,1+len(hwords)):
        wn = list(ngrams(words,N))
        hn = list(ngrams(hwords,N))
        cm = filter(lambda x: x in wn, hn)
        bleus += (len(cm)+0.01) / len(hn)
    bleus /= len(hwords)
    return bleus;

def feature_neg_match(sentence_A_tokens,sentence_B_tokens,sentence_A_pos,sentence_B_pos):
	feature_neg_match=0;
	for neg_word in neg_words:
		if neg_word in sentence_B_tokens and neg_word in sentence_A_tokens:
				feature_neg_match+=1;
		elif neg_word in sentence_B_tokens and neg_word not in sentence_A_tokens:
				feature_neg_match-=1;
		elif neg_word not in sentence_B_tokens and neg_word in sentence_A_tokens:
				feature_neg_match-=1;
	feature_neg_match/=min(len(sentence_A_tokens),len(sentence_B_tokens))
	return feature_neg_match


def feature_num_match(sentence_A_tokens,sentence_B_tokens,sentence_A_pos,sentence_B_pos):
	sentence_B_pos_nums=[words for (words,pos) in sentence_B_pos if pos=='CD'];
	sentence_A_pos_nums=[words for (words,pos) in sentence_A_pos if pos=='CD'];
	feature_num_match=0;
	for w_a in sentence_B_pos_nums:
			cd_present=0;
			for w_e in sentence_A_pos_nums:
					cd_present=1;
					if w_e==w_a:
						feature_num_match+=1;
						cd_present=0;
						break;
			if cd_present==1:
				feature_num_match-=1;
	feature_num_match/=min(len(sentence_A_tokens),len(sentence_B_tokens))
	return feature_num_match


#Antonyn and Synonym features
def append_antonym_synonym_wup(sentence_A_tokens,sentence_B_tokens,sentence_A_pos,sentence_B_pos, feature_row):
	antonym_match=0;
	synonym_match=0;
	
	wup_cumulative=0;
	
	for (word,pos) in sentence_B_pos:
		max_wup=0;
		word_l=lmtzr.lemmatize(word);
		#word_l=word
		for(word_e,pos_e) in sentence_A_pos:
			word_e_l=lmtzr.lemmatize(word_e);
			#word_e_l=word_e
			try:
				max_wup=max(max_wup,wn.synsets(word_e_l)[0].wup_similarity(wn.synsets(word_l)[0]));
			except IndexError:
				pass
		wup_cumulative+=max_wup;
		if pos in pos_list:
			word_synsets=wn.synsets(word_l);
			synonyms=list(set([lemma2.key.split('%')[0] for lemma2 in [t.lemmas[0] for t in word_synsets]]));
			antonyms=list(set([lemma2.key.split('%')[0] for lemma2 in [t.lemmas[0].antonyms()[0] for t in word_synsets if t.lemmas[0].antonyms()]]));
			for (word_e,pos_e) in sentence_A_pos:
				word_e_l=lmtzr.lemmatize(word_e);
				if word_e_l in synonyms:
					#check whether not is there or not
					if sentence_A_tokens.index(word_e)!=0 and sentence_B_tokens.index(word)!=0:
						if (sentence_A_tokens[sentence_A_tokens.index(word_e)-1] in neg_words and sentence_B_tokens[sentence_B_tokens.index(word)-1] in neg_words) or (sentence_A_tokens[sentence_A_tokens.index(word_e)-1] not in neg_words and sentence_B_tokens[sentence_B_tokens.index(word)-1] not in neg_words):
							if pos_dict2[pos]=='NOUN':
								synonym_match+=0.5
							elif pos_dict2[pos]=='VERB':
								synonym_match+=0.3
							elif pos_dict2[pos]=='ADJ':
								synonym_match+=0.1
							elif pos_dict2[pos]=='ADV':
								synonym_match+=0.1
						else: 
							synonym_match-=0.5;

				elif word_e in antonyms:
					#check whether not is there or not 
					if sentence_A_tokens.index(word_e)!=0 and sentence_B_tokens.index(word)!=0:
						if (sentence_A_tokens[sentence_A_tokens.index(word_e)-1] in neg_words and sentence_B_tokens[sentence_B_tokens.index(word)-1] in neg_words) or (sentence_A_tokens[sentence_A_tokens.index(word_e)-1] not in neg_words and sentence_B_tokens[sentence_B_tokens.index(word)-1] not in neg_words):
							if pos_dict2[pos]=='NOUN':
								antonym_match+=0.5
							elif pos_dict2[pos]=='VERB':
								antonym_match+=0.3
							elif pos_dict2[pos]=='ADJ':
								antonym_match+=0.1
							elif pos_dict2[pos]=='ADV':
								antonym_match+=0.1
						else: 
							antonym_match-=0.5;
	synonym_match=synonym_match/min(len(sentence_A_tokens),len(sentence_B_tokens))
	antonym_match=antonym_match/min(len(sentence_A_tokens),len(sentence_B_tokens))
	wup_cumulative=wup_cumulative/min(len(sentence_A_tokens),len(sentence_B_tokens))
	feature_row.append(synonym_match)
	feature_row.append(antonym_match)
	feature_row.append(wup_cumulative)
	return feature_row

def overlap(sentence_A_tokens,sentence_B_tokens):
	s1=set(sentence_A_tokens)
	s2=set(sentence_B_tokens)
	if max(len(s1),len(s2))==0:
		overlap=0
	else:
		overlap=2*len(s1&s2)/len(s1|s2)
	return overlap

def overlap_neg(sentence_A_tokens,sentence_B_tokens):
	s1=set(sentence_A_tokens)
	s2=set(sentence_B_tokens)
	if max(len(s1),len(s2))==0:
		overlap=0
	else:
		if (s1^s2)&set(neg_words):
			overlap=2*len((s1^s2)&set(neg_words))/len(s1|s2)
		else:
			overlap=0 
	return overlap

def max_match(matrix):
	cost_matrix = make_cost_matrix(matrix, lambda cost: 100.0-cost)
	m = Munkres()
	indexes = m.compute(cost_matrix)
	#print_matrix(matrix, msg='Lowest cost through this matrix:')
	total = 0
	for row, column in indexes:
	    value = matrix[row][column]
	    total += value
	    #print '(%d, %d) -> %d' % (row, column, value)
	return total

def feature_bipartite_matching2(sentence_A_tokens,sentence_B_tokens,sentence_A_pos, sentence_B_pos, feature_row):
	graph=[[0.0 for x in xrange(len(sentence_B_tokens))] for y in xrange(len(sentence_A_tokens))]
	for i in xrange(len(sentence_A_tokens)):
		(word_A_i,pos_A_i)=sentence_A_pos[i]
		for j in xrange(len(sentence_B_tokens)):
			(word_B_j,pos_B_j)=sentence_B_pos[j]

			if pos_A_i==pos_B_j:
				try:
					res1=wn.synsets(word_A_i,pos_dict[pos_A_i])[0].res_similarity(wn.synsets(word_B_j,pos_dict[pos_B_j])[0], brown_ic)
					if res1!=None and res1<100:
						graph[i][j]=res1
				except:
					pass
	feature_row.append(max_match(graph)/min(len(sentence_B_tokens),len(sentence_A_tokens)))
	return feature_row


def feature_bipartite_matching(sentence_A_tokens,sentence_B_tokens,sentence_A_pos, sentence_B_pos, feature_row):
	graph=[[0.0 for x in xrange(len(sentence_B_tokens))] for y in xrange(len(sentence_A_tokens))]
	for i in xrange(len(sentence_A_tokens)):
		(word_A_i,pos_A_i)=sentence_A_pos[i]
		for j in xrange(len(sentence_B_tokens)):
			(word_B_j,pos_B_j)=sentence_B_pos[j]

			if pos_A_i==pos_B_j:
				#print pos_A_i+" "+pos_B_j
				try:
					res1=wn.synsets(word_A_i,pos_dict[pos_A_i])[0].res_similarity(wn.synsets(word_B_j,pos_dict[pos_B_j])[0], brown_ic)
					#res2=wn.synsets(word_A_i)[0].wup_similarity(wn.synsets(word_B_j)[0], semcor_ic)
					if res1!=None and res1<100:
						graph[i][j]=res1
					#if res2!=None:
					#	graph_semcor_ic[i][j]=res2
				except:
					try:
						wup=wn.synsets(word_A_i)[0].wup_similarity(wn.synsets(word_B_j)[0])
						if wup!= None:
							graph[i][j]=wup
					except IndexError:
						pass
			else:
				#print pos_A_i+" "+pos_B_j
				try:
					wup=wn.synsets(word_A_i)[0].wup_similarity(wn.synsets(word_B_j)[0])
					if wup!= None and wup<100:
						graph[i][j]=wup
						#debug
						#print str(i)+" "+str(j)+" "+str(wup)+" "+str(graph[i][j])	
				except IndexError:
					pass
	#print graph
	
	feature_row.append(max_match(graph)/min(len(sentence_B_tokens),len(sentence_A_tokens)))
	#feature_row.append(max_match(graph_brown_ic))
	#feature_row.append(max_match(graph_semcor_ic))
	return feature_row

def compute_c_features(sentence_A,sentence_B):
	feature_row=[];
	#1
	feature_row.append(bleu(sentence_A,sentence_B))
	
	sentence_A_tokens=nltk.word_tokenize(sentence_A);
	sentence_B_tokens=nltk.word_tokenize(sentence_B);
	sentence_B_pos=nltk.pos_tag(sentence_B_tokens);
	sentence_A_pos=nltk.pos_tag(sentence_A_tokens);
	#2-10
	feature_row.append(feature_neg_match(sentence_A_tokens,sentence_B_tokens,sentence_A_pos,sentence_B_pos));
	feature_row.append(feature_num_match(sentence_A_tokens,sentence_B_tokens,sentence_A_pos,sentence_B_pos));
	feature_row=append_antonym_synonym_wup(sentence_A_tokens,sentence_B_tokens,sentence_A_pos,sentence_B_pos, feature_row)
	feature_row.append(overlap(sentence_A_tokens,sentence_B_tokens))
	feature_row=feature_bipartite_matching(sentence_A_tokens,sentence_B_tokens,sentence_A_pos, sentence_B_pos, feature_row)
	feature_row=feature_bipartite_matching2(sentence_A_tokens,sentence_B_tokens,sentence_A_pos, sentence_B_pos, feature_row)
	#feature_row.append(overlap_neg(sentence_A_tokens, sentence_B_tokens))
	#11-
	feature_row.extend(calc_c_features(sentence_A_pos,sentence_B_pos))
	
	#feature_row.append(1)
	return feature_row;

def compute_r_features(sentence_A,sentence_B):
	feature_row=[];
	#1
	feature_row.append(bleu(sentence_A,sentence_B))
	
	sentence_A_tokens=nltk.word_tokenize(sentence_A);
	sentence_B_tokens=nltk.word_tokenize(sentence_B);
	sentence_B_pos=nltk.pos_tag(sentence_B_tokens);
	sentence_A_pos=nltk.pos_tag(sentence_A_tokens);
	#2-10
	feature_row.append(feature_neg_match(sentence_A_tokens,sentence_B_tokens,sentence_A_pos,sentence_B_pos));
	feature_row.append(feature_num_match(sentence_A_tokens,sentence_B_tokens,sentence_A_pos,sentence_B_pos));
	feature_row=append_antonym_synonym_wup(sentence_A_tokens,sentence_B_tokens,sentence_A_pos,sentence_B_pos, feature_row)
	feature_row.append(overlap(sentence_A_tokens,sentence_B_tokens))
	feature_row=feature_bipartite_matching(sentence_A_tokens,sentence_B_tokens,sentence_A_pos, sentence_B_pos, feature_row)
	feature_row=feature_bipartite_matching2(sentence_A_tokens,sentence_B_tokens,sentence_A_pos, sentence_B_pos, feature_row)
	#feature_row.append(overlap_neg(sentence_A_tokens, sentence_B_tokens))
	#11-
	feature_row.extend(calc_r_features(sentence_A_pos,sentence_B_pos))
	
	#feature_row.append(1)
	return feature_row;

if __name__ == '__main__':
	#checking
	sentence_A="The white dog wearing reindeer ears is close to a brown dog"
	sentence_B="A white dog is wearing a Christmas reindeer headband and is playing with a brown dog in the grass"
	sentence_A_tokens=nltk.word_tokenize(sentence_A);
	sentence_B_tokens=nltk.word_tokenize(sentence_B);
	sentence_B_pos=nltk.pos_tag(sentence_B_tokens);
	sentence_A_pos=nltk.pos_tag(sentence_A_tokens);
	feature_row=[];
	print feature_bipartite_matching(sentence_A_tokens,sentence_B_tokens, sentence_A_pos,sentence_B_pos, feature_row)
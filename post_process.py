def final_score(pred_score):

	if pred_score>5:
		return 5
	elif pred_score<0:
		return 0
	else:
		return round(pred_score,2)
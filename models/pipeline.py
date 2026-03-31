from transformers import pipeline
import numpy as np
model=pipeline("sentiment-analysis", truncation=True )

#sentiment analysis for the reviews 
def predict(reviews):
    analysis_scores=[]
    if reviews :
            for review in reviews:
                res=model(review)[0]
                if res["label"]=="POSITIVE":
                    score=res["score"]
                else:
                    score=1-res["score"]
                analysis_scores.append(score)
     
    avg_score=np.mean(analysis_scores) if analysis_scores else 0
    return avg_score
def overview_analysis(overview):
    
    result = model(overview)[0]
    if result['label'] == 'POSITIVE':
        return result['score']
    return 1 - result['score']
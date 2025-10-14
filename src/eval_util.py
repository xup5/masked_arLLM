
def compute_openQA_recall(pred, gt):
    pred, gt = pred.strip(), gt.strip()
    
    import string
    from rouge import Rouge
    import nltk
    from nltk import word_tokenize
    nltk.download('punkt')
    nltk.download('punkt_tab')
    
    exclude_voc = [ "a", "of", "the", "and", "is", "are", "be", "to", "new", "on", "in", "at", "an", "for"]

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    

    pred = remove_punc(pred)
    tokenized_pred = word_tokenize(pred)
    tokenized_pred = [w for w in tokenized_pred if w.lower() not in exclude_voc]
    pred = ' '.join(tokenized_pred)


    if isinstance(gt, str):
        gts = [gt]
    else:
        gts = gt
    gts = [remove_punc(s) for s in gts]
    tokenized_refs  = [word_tokenize(s) for s in gts]
    tokenized_refs  = [ [w for w in tokenized_ref  if w.lower() not in exclude_voc] for tokenized_ref in tokenized_refs]
    gts = [' '.join(s) for s in tokenized_refs]

    rouge = Rouge()
    rouge_scores = rouge.get_scores(hyps=[pred]*len(gts), refs=gts)
    recall = max([rouge_scores[i]["rouge-1"]["r"] for i in range(len(rouge_scores))])
    
    return recall

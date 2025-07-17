from transformers import pipeline
from bert_score import BERTScorer

def evaluate(explanation, gt):
    # 1. Relevance evaluation with BERTScorer
    scorer = BERTScorer(model_type='bert-base-uncased')
    P, R, F1 = scorer.score([explanation], [gt], verbose=True)
    print(f'Statement: {gt}')
    print(
        f'P: {float(P.detach().numpy().squeeze()):.3f}, R: {float(R.detach().numpy().squeeze()):.3f}, F1: {float(F1.detach().numpy().squeeze()):.3f}')

    # 2. Alignment with ROBERTa NLI module
    nli = pipeline("text-classification", model="roberta-large-mnli", truncation=True)
    result = nli({"text": explanation, "text_pair": gt})
    print(result)

    return
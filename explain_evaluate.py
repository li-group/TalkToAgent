from transformers import pipeline
from bert_score import BERTScorer

true_statements = [
    "The counterfactual trajectory showed overshooting behavior in h1 state.",
    "The counterfactual trajectory did not show overshooting behavior in h2 state.",
    "The counterfactual trajectory showed undershooting behavior in h2 state.",
    "The actual trajectory settled faster than the counterfactual trajectory.",
    "The counterfactual was better in control performance than the original trajectory"
]

false_statements = [
    "The counterfactual trajectory did not show overshooting behavior in h1 state.",
    "The counterfactual trajectory showed overshooting behavior in h2 state.",
    "The counterfactual trajectory did not show undershooting behavior in h2 state.",
    "The counterfactual trajectory settled faster than the actual trajectory.",
    "The counterfactual was worse in control performance than the original trajectory"
]

def evaluate(explanation, true_statements, false_statements):
    true_results = []
    false_results = []

    for st in true_statements:
        # 1. Relevance evaluation with BERTScorer
        scorer = BERTScorer(model_type='bert-base-uncased')
        P, R, F1 = scorer.score([explanation], [gt], verbose=True)
        print(f'Statement: {st}')
        print(
            f'P: {float(P.detach().numpy().squeeze()):.3f}, R: {float(R.detach().numpy().squeeze()):.3f}, F1: {float(F1.detach().numpy().squeeze()):.3f}')

        # 2. Alignment with ROBERTa NLI module
        nli = pipeline("text-classification", model="roberta-large-mnli", truncation=True)
        result = nli({"text": explanation, "text_pair": gt})
        print(result)
        true_results.append(result)

    for st in false_statements:
        # 1. Relevance evaluation with BERTScorer
        scorer = BERTScorer(model_type='bert-base-uncased')
        P, R, F1 = scorer.score([explanation], [gt], verbose=True)
        print(f'Statement: {st}')
        print(
            f'P: {float(P.detach().numpy().squeeze()):.3f}, R: {float(R.detach().numpy().squeeze()):.3f}, F1: {float(F1.detach().numpy().squeeze()):.3f}')

        # 2. Alignment with ROBERTa NLI module
        nli = pipeline("text-classification", model="roberta-large-mnli", truncation=True)
        result = nli({"text": explanation, "text_pair": gt})
        print(result)
        false_results.append(result)
# ANLP Exercise 2 – Part 2 | Netanel Azran
#
# Failure type: contrast / expectation-reversal sentences.
# The model sees strong early sentiment (positive or negative) and ignores
# the final clause that reverses the overall sentiment.
#
# Hypothesis: models over-weight emotionally salient early tokens and
# under-weight the concluding evaluative clause.
#
# Hypothesis tests probe: word order, conjunction choice, adjective intensity,
# negation, and the presence/absence of a contrasting clause.

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

pd.set_option("display.max_colwidth", None)

MODEL_1 = "Qwen/Qwen3-8B"
MODEL_2 = "meta-llama/Llama-3.1-8B-Instruct"

SYSTEM = ("You are a sentiment classifier. Decide the overall sentiment of the text. "
          "Answer with one word, Positive or Negative, and a confidence in parentheses, "
          "like 'Positive (very confident)'. Do not explain.")

# Break examples: strong early sentiment contradicted by the final verdict
break_examples = [
    ("The cinematography was absolutely stunning and the score was breathtaking - none of which saved this film from being a tedious, lifeless bore.", "Negative"),
    ("What a phenomenal cast and a gorgeous set; too bad the script was so insulting that I walked out before the halfway mark.", "Negative"),
    ("I adored every glowing review I read about this restaurant. The actual meal was cold, overpriced, and completely forgettable.", "Negative"),
    ("Brilliant premise, an award-worthy trailer, an A-list director - and yet the movie itself is two hours of pure boredom.", "Negative"),
    ("The hotel website promised a luxurious spa and a five-star breakfast. In reality the room was filthy and the staff were rude.", "Negative"),
    ("The trailer looked like a complete disaster and the early reviews were brutal, but honestly this turned out to be the most moving film I have seen all year.", "Positive"),
]

# Hypothesis tests: vary order, conjunction, intensity, negation, and control
hypothesis_tests = [
    # H1 – flip order: negative first, positive second → should still be Negative overall
    ("This film was a tedious, lifeless bore, even though the cinematography was stunning and the score was breathtaking.", "Negative"),
    # H2 – explicit adversative conjunction "despite"
    ("Despite stunning cinematography and a breathtaking score, the film was a tedious, lifeless bore.", "Negative"),
    # H3 – weaker positive words (fine/okay) to see if intensity drives the error
    ("The cinematography was fine and the score was okay, but the film was a tedious, lifeless bore.", "Negative"),
    # H4 – amplify the positive superlatives even further
    ("The cinematography was the most jaw-dropping, gorgeous, breathtaking work I have ever seen - yet the film was a tedious, lifeless bore.", "Negative"),
    # H5 – simple negative, no contrast (control)
    ("The movie was not great and the acting was not good.", "Negative"),
    # H6 – negation reversal: "not the worst" → actually positive
    ("It is not the worst film ever; in fact it was genuinely wonderful.", "Positive"),
    # H7 – pure negative, no positive words (control)
    ("A boring, lifeless, tedious film that I regret watching.", "Negative"),
    # H8 – pure positive, no negative words (control)
    ("A stunning, breathtaking film that I absolutely loved.", "Positive"),
]


def load_model(name):
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    return tok, model


def classify(tok, model, text, name):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]
    if "qwen3" in name.lower():
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    else:
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=24, do_sample=False, pad_token_id=tok.eos_token_id)
    answer = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return answer.strip().replace("\n", " ")


def run(tok, model, name, items):
    rows = []
    for i, (text, gold) in enumerate(items, 1):
        ans = classify(tok, model, text, name)
        print(i, "|", gold, "->", ans)
        rows.append({"Index": i, "Input": text, "Gold": gold, "Model output": ans})
    return pd.DataFrame(rows)


print("loading", MODEL_1)
tok1, model1 = load_model(MODEL_1)

print("\n=== break the model (model 1) ===")
t1 = run(tok1, model1, MODEL_1, break_examples)
print("\n=== hypothesis tests (model 1) ===")
t2 = run(tok1, model1, MODEL_1, hypothesis_tests)

del model1
torch.cuda.empty_cache()

print("\nloading", MODEL_2)
tok2, model2 = load_model(MODEL_2)

print("\n=== break the model (model 2) ===")
t3 = run(tok2, model2, MODEL_2, break_examples)
print("\n=== hypothesis tests (model 2) ===")
t4 = run(tok2, model2, MODEL_2, hypothesis_tests)

t1.to_csv("table1_break_model1.csv", index=False)
t2.to_csv("table2_hyp_model1.csv", index=False)
t3.to_csv("table3_break_model2.csv", index=False)
t4.to_csv("table4_hyp_model2.csv", index=False)
print("\ndone, saved csv files")

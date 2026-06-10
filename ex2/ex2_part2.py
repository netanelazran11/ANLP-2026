# ANLP Exercise 2 – Part 2 | Netanel Azran
#
# Failure type: implicit negative sentiment — sentences that are negative but
# contain NO explicit negative words. The negativity must be inferred from
# understatement, irony, or real-world consequence.
#
# Hypothesis: the model classifies sentiment from the presence of explicit
# positive/negative vocabulary. When negativity is only implied (surface words
# are neutral or positive), the model defaults to Positive.
#
# Hypothesis tests: controlled variation on a single scenario (umbrella).
# We change ONLY how explicit the negativity is, keeping everything else fixed.
# Expected pattern: model correct on explicit versions, wrong on understatement.
#
# Proposed solution: two-step prompting — first ask the model to describe the
# implied outcome ("what does this say about the product?"), then classify
# that generated text, which will contain explicit polarity.

import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

pd.set_option("display.max_colwidth", None)

MODEL_1 = "Qwen/Qwen3-8B"
MODEL_2 = "meta-llama/Llama-3.1-8B-Instruct"

SYSTEM = ("You are a sentiment classifier. Decide the overall sentiment of the text. "
          "Answer with one word, Positive or Negative, and a confidence in parentheses, "
          "like 'Positive (very confident)'. Do not explain.")

# Break examples: negative sentiment implied through understatement or
# real-world consequence — zero explicit negative words on the surface.
break_examples = [
    ("This umbrella kept me perfectly dry for the first ten seconds of the storm.", "Negative"),
    ("The battery lasts just long enough to get from my bedroom to the kitchen.", "Negative"),
    ("Customer support got back to me in only three hours, which was faster than my last ticket.", "Negative"),
    ("The phone survived a drop from pocket height, as long as I was sitting down at the time.", "Negative"),
    ("After only two visits to the mechanic this month, the car is running perfectly again.", "Negative"),
    ("The noise-cancelling headphones reduced the drilling noise next door to a very manageable level.", "Negative"),
    ("I have now successfully assembled three of the seven required pieces.", "Negative"),
    ("The waterproofing performed admirably in light drizzle, the only condition I had the chance to test.", "Negative"),
    ("I got to sample the entire menu, as each dish came back to the kitchen at least once.", "Negative"),
]

# Hypothesis tests: one scenario (umbrella), vary ONLY how explicit the negativity is.
# H1–H3: explicit negatives  → model should be correct (Negative).
# H4–H6: pure understatement → model should FAIL (predict Positive).
# H7–H8: positive controls   → model should be correct (Positive).
hypothesis_tests = [
    # H1 – fully explicit negative (control)
    ("This umbrella broke in the first minute of rain and I got completely soaked.", "Negative"),
    # H2 – explicit negative with opinion word
    ("This umbrella is useless. It fell apart immediately and left me drenched.", "Negative"),
    # H3 – sarcasm marker present ("Oh great")
    ("Oh great, another umbrella that lasts all of ten seconds in the rain.", "Negative"),
    # H4 – pure understatement, zero negative words → expected model failure
    ("This umbrella kept me dry for the first ten seconds of the storm.", "Negative"),
    # H5 – understatement with a specific number → expected model failure
    ("The battery lasts just long enough to reach the coffee machine from my desk.", "Negative"),
    # H6 – implied consequence, no sentiment words → expected model failure
    ("I used this umbrella once before going back to my old one.", "Negative"),
    # H7 – explicit positive (control)
    ("This umbrella is fantastic. It kept me completely dry through two hours of heavy rain.", "Positive"),
    # H8 – positive understatement (control): model should still be correct
    ("This umbrella handled a two-hour downpour without a single leak.", "Positive"),
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

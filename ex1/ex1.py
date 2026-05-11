import argparse
import numpy as np
import wandb
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
import evaluate

MODEL_NAME = "bert-base-uncased"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--max_eval_samples", type=int, default=-1)
    parser.add_argument("--max_predict_samples", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--do_train", action="store_true")
    parser.add_argument("--do_predict", action="store_true")
    parser.add_argument("--model_path", type=str, default=None)
    return parser.parse_args()


def load_and_tokenize(tokenizer):
    dataset = load_dataset("glue", "mrpc")

    def tokenize(examples):
        return tokenizer(
            examples["sentence1"],
            examples["sentence2"],
            truncation=True,
            max_length=tokenizer.model_max_length,
        )

    return dataset.map(tokenize, batched=True), dataset


def compute_metrics(eval_pred):
    metric = evaluate.load("accuracy")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenized, raw_dataset = load_and_tokenize(tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = tokenized["train"]
    eval_dataset = tokenized["validation"]
    test_dataset = tokenized["test"]

    if args.max_train_samples != -1:
        train_dataset = train_dataset.select(range(args.max_train_samples))
    if args.max_eval_samples != -1:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))
    if args.max_predict_samples != -1:
        test_dataset = test_dataset.select(range(args.max_predict_samples))

    if args.do_train:
        run_name = f"bert-mrpc-ep{args.num_train_epochs}-lr{args.lr}-bs{args.batch_size}"
        output_dir = f"./results/{run_name}"

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=args.num_train_epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            save_total_limit=1,
            logging_steps=1,
            report_to="wandb",
            run_name=run_name,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        eval_results = trainer.evaluate()
        val_acc = eval_results["eval_accuracy"]

        with open("res.txt", "a") as f:
            f.write(
                f"epoch_num: {args.num_train_epochs}, lr: {args.lr}, "
                f"batch_size: {args.batch_size}, eval_acc: {val_acc:.4f}\n"
            )

        trainer.save_model(output_dir + "/best_model")
        print(f"Model saved to {output_dir}/best_model")
        wandb.finish()

    if args.do_predict:
        assert args.model_path is not None, "--model_path is required for prediction"

        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)

        training_args = TrainingArguments(
            output_dir="./predict_tmp",
            per_device_eval_batch_size=args.batch_size,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            processing_class=tokenizer,
            data_collator=data_collator,
        )

        model.eval()
        predictions = trainer.predict(test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=-1)

        raw_test = raw_dataset["test"]
        if args.max_predict_samples != -1:
            raw_test = raw_test.select(range(args.max_predict_samples))

        with open("predictions.txt", "w") as f:
            for i, label in enumerate(pred_labels):
                s1 = raw_test[i]["sentence1"]
                s2 = raw_test[i]["sentence2"]
                f.write(f"{s1}###{s2}###{label}\n")

        print("Predictions saved to predictions.txt")


if __name__ == "__main__":
    main()

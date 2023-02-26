import os
from tqdm import tqdm

from transformers import (
    get_scheduler,
    AutoTokenizer,
    DataCollatorWithPadding,
    AutoModelForSequenceClassification,
    AdamW,
)
import datasets

import torch
from torch.utils.data import DataLoader

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score
)

class DetectorPipeline:
    def __init__(
            self,
            dataset_name: str = "GonzaloA/fake_news",
            checkpoint: str = "distilbert-base-uncased-finetuned-sst-2-english",
            model_name: str = "fake_news_detector"
        ):
        """Detector pipeline class.

        :param dataset_name: Name of the dataset to download, defaults to "GonzaloA/fake_news"
        :type dataset_name: str, optional
        :param checkpoint: Name of the model to fine-tune, defaults to "distilbert-base-uncased-finetuned-sst-2-english"
        :type checkpoint: str, optional
        :param model_name: Name of the model to save, defaults to "fake_news_detector"
        :type model_name: str, optional
        """
        self.dataset_name = dataset_name
        self.checkpoint = checkpoint
        self.model_name = model_name

    def download_dataset(self) -> datasets.DatasetDict:
        """Download dataset from HuggingFace datasets library.

        :return: DatasetDict object with the train, validation and test splits.
        :rtype: datasets.DatasetDict
        """

        dataset = datasets.load_dataset(self.dataset_name)
        return dataset

    def get_tokenizer_and_model(self) -> (AutoTokenizer, AutoModelForSequenceClassification):
        """Get tokenizer and model from model name.

        :return: Tokenizer and Model objects.
        :rtype: (AutoTokenizer, AutoModelForSequenceClassification)
        """

        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        model = AutoModelForSequenceClassification.from_pretrained(self.checkpoint, num_labels=2)
        return tokenizer, model

    def get_data_collator(self, tokenizer: AutoTokenizer) -> DataCollatorWithPadding:
        """Get data collator from tokenizer.

        :param tokenizer: Tokenizer object.
        :type tokenizer: AutoTokenizer
        :return: Data collator object.
        :rtype: DataCollatorWithPadding
        """

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        return data_collator

    def get_dataloaders(self, dataset: datasets.DatasetDict, tokenizer: AutoTokenizer, data_collator: DataCollatorWithPadding) -> (DataLoader, DataLoader, DataLoader):
        """_summary_

        :param dataset: Train dataset.
        :type dataset: datasets.DatasetDict
        :param DataLoader: Eval dataset.
        :type DataLoader: datasets.DatasetDict
        :param DataLoader: Test dataset.
        :type DataLoader: datasets.DatasetDict
        """

        # Tokenize dataset
        def tokenize_function(example):
            return tokenizer(example["text"], truncation=True, padding=True)
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)

        # Put in format that the model expects
        tokenized_dataset = tokenized_dataset.remove_columns(["Unnamed: 0", "title", "text"])
        tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
        tokenized_dataset.set_format("torch")

        # Create dataloaders
        train_dataloader = DataLoader(
            tokenized_dataset["train"], shuffle=True, batch_size=32, collate_fn=data_collator
        )
        eval_dataloader = DataLoader(
            tokenized_dataset["validation"], batch_size=32, collate_fn=data_collator
        )
        test_dataloader = DataLoader(
            tokenized_dataset["test"], batch_size=32, collate_fn=data_collator
        )
        return train_dataloader, eval_dataloader, test_dataloader

    def train_model(self, model: AutoModelForSequenceClassification, train_dataloader: DataLoader, epochs: int = 3, lr: float = 2e-5, weight_decay: float = 0.0, warmup_steps: int = 0, max_grad_norm: float = 1.0) -> AutoModelForSequenceClassification:
        """Train model.

        :param model: Model to train.
        :type model: AutoModelForSequenceClassification
        :param train_dataloader: Dataloader with train data.
        :type train_dataloader: DataLoader
        :param epochs: Number of epochs to train, defaults to 3
        :type epochs: int, optional
        :param lr: Learning rate, defaults to 2e-5
        :type lr: float, optional
        :param weight_decay: Weight decay, defaults to 0.0
        :type weight_decay: float, optional
        :param warmup_steps: Number of warmup steps, defaults to 0
        :type warmup_steps: int, optional
        :param max_grad_norm: Maximum gradient norm, defaults to 1.0
        :type max_grad_norm: float, optional
        :return: Trained model.
        :rtype: AutoModelForSequenceClassification
        """

        num_training_steps = len(train_dataloader) * epochs

        # Set device
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        # Set optimizer
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        # Set scheduler
        scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Train
        pbar = tqdm(range(num_training_steps))
        with tqdm(total=num_training_steps) as pbar:
            for epoch in range(epochs):
                model.train()
                for i, batch in enumerate(train_dataloader):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    loss = outputs.loss
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    pbar.write(f'processed: {i * (epoch + 1) / num_training_steps * 100:.2f}%')
                    pbar.update(1)

        return model

    def evaluate_model(self, dataset: datasets.Dataset, dataloader: DataLoader, model: AutoModelForSequenceClassification) -> float:
        """Evaluate model.

        :param eval_dataloader: dataset to evaluate.
        :type eval_dataloader: DatasetDict
        :param dataset: Dataloader with eval data.
        :type dataset: DataLoader
        :param model: Model to evaluate.
        :type model: AutoModelForSequenceClassification
        :return: Accuracy.
        :rtype: float
        """

        # Set device
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        # Evaluate
        model.eval()
        predictions = []

        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            logits = outputs.logits
            running_predictions = torch.argmax(logits, dim=-1).to('cpu').numpy().tolist()
            predictions.extend(running_predictions)

        print("Results of the model:\n")
        f1score = f1_score(dataset["validation"]["labels"], predictions, average="macro")
        print(f"F1 score: {f1score}")
        print(classification_report(dataset["validation"]["labels"], predictions))
        print(confusion_matrix(dataset["validation"]["labels"], predictions))

        return f1score
    
    def train_pipeline(self) -> AutoModelForSequenceClassification:
        """Runs the train pipeline and returns the trained model.

        :return: Trained model.
        :rtype: AutoModelForSequenceClassification
        """
        dataset = self.load_dataset()
        train_dataloader, eval_dataloader, test_dataloader = self.tokenize_dataset(dataset)
        model = self.load_model()
        model = self.train_model(model, train_dataloader)
        self.evaluate_model(dataset, eval_dataloader, model)
        return model

    def load_model_from_directory(self, model_name: str = None) -> AutoModelForSequenceClassification:
        """Load model from directory.

        :param model_name: Name of the model to load, if None, self.model_name is used, defaults to None.
        :type model_name: str
        :return: Loaded model.
        :rtype: AutoModelForSequenceClassification
        """
        load_path = os.path.join("models", model_name) if model_name else os.path.join("models", self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(load_path, local_files_only=True)
        return model

    def predict(self, model: AutoModelForSequenceClassification, text: str) -> int:
        """Predict class of text.

        :param model: Model to use for prediction.
        :type model: AutoModelForSequenceClassification
        :param text: Text to predict.
        :type text: str
        :return: Predicted class.
        :rtype: int
        """
        # Set device
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model.to(device)

        # Predict
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(self.checkpoint)
        encoded_text = tokenizer(text, truncation=True, padding=True, return_tensors="pt")
        encoded_text = {k: v.to(device) for k, v in encoded_text.items()}
        with torch.no_grad():
            outputs = model(**encoded_text)
        logits = outputs.logits
        prediction = torch.argmax(logits, dim=-1).to('cpu').numpy().tolist()[0]
        return prediction

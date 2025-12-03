"""Integration of AraBERT for Fake News Detection.
Provides a transformer-based classifier with optional fine-tuning.
Requires optional dependencies: torch, transformers.
"""

from typing import List, Optional
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel

from src.utils import logger, save_model, load_model

DEFAULT_MODEL_NAME = "aubmindlab/bert-base-arabertv2"

class AraBERTFakeNewsClassifier(nn.Module):
    def __init__(self, model_name: str = DEFAULT_MODEL_NAME, num_labels: int = 2):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.base_model = AutoModel.from_pretrained(model_name)
        hidden_size = self.base_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(hidden_size, num_labels)
        )
        self.to(self.device)
        self.num_labels = num_labels

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask)
        # Use CLS token representation
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits

    def predict(self, texts: List[str]):
        self.eval()
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = probs.argmax(axis=1)
        return preds, probs

    def fine_tune(self, texts: List[str], labels: List[int], epochs: int = 2, batch_size: int = 8, lr: float = 2e-5):
        self.train()
        encoded = self.tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors='pt')
        dataset = torch.utils.data.TensorDataset(encoded['input_ids'], encoded['attention_mask'], torch.tensor(labels))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in loader:
                input_ids, attention_mask, y = [t.to(self.device) for t in batch]
                optimizer.zero_grad()
                logits = self.forward(input_ids, attention_mask)
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(loader):.4f}")
        logger.info("Fine-tuning completed")

    def save(self, path: str):
        save_model(self, path)

    @staticmethod
    def load(path: str):
        model = load_model(path)
        return model

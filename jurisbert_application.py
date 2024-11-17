import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.model_selection import KFold
from datasets import Dataset
import pandas as pd

#carregar df
df = balanced_df.copy()
df = df[['ementa', 'favoravel']]
label_mapping = {'banco': 0, 'cliente': 1}
df['labels'] = df['favoravel'].map(label_mapping)

#carregar modelo e tokenizador
model_name = "alfaneo/jurisbert-base-portuguese-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize(batch):
    return tokenizer(batch['ementa'], padding='max_length', truncation=True, max_length=512)

#calcular metricas
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

#plotar matriz de confusão individual
def plot_confusion_matrix(y_true, y_pred, fold_number=None,vmin=0, vmax=35):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis',
                xticklabels=['Banco', 'Cliente'], yticklabels=['Banco', 'Cliente'],
                vmin=vmin, vmax=vmax)
    plt.xlabel('Predito')
    plt.ylabel('Real')
    title = f'Matriz de Confusão - Validação (Fold {fold_number})' if fold_number is not None else 'Matriz de Confusão Final'
    plt.title(title)
    plt.show()

#plotar todas as matrizes juntas
def plot_combined_confusion_matrices(results_list):
    num_folds = len(results_list)
    fig, axes = plt.subplots(1, num_folds, figsize=(6 * num_folds, 4))

    for i, (y_true, y_pred) in enumerate(results_list):
        conf_matrix = confusion_matrix(y_true, y_pred)
        ax = axes[i] if num_folds > 1 else axes
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis',
                    xticklabels=['Banco', 'Cliente'], yticklabels=['Banco', 'Cliente'], ax=ax)
        ax.set_xlabel('Predito')
        ax.set_ylabel('Real')
        ax.set_title(f'Fold {i + 1}')

    plt.suptitle('Matrizes de Confusão - Todos os Folds')
    plt.show()

#preparar KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results_list = []

#executar o KFold
for fold_number, (train_index, test_index) in enumerate(kf.split(df)):
    train_df = df.iloc[train_index]
    test_df = df.iloc[test_index]

    #tokenizar datasets
    train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    test_dataset = Dataset.from_pandas(test_df).map(tokenize, batched=True)

    #carregar modelo
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    #configurar o treinamento
    training_args = TrainingArguments(
        output_dir='/content/drive/MyDrive/TCC_FINAL/classified_data/treinamento',
        num_train_epochs=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        logging_dir='/content/drive/MyDrive/TCC_FINAL/classified_data/logs',
        logging_steps=10,
        learning_rate=5e-5,
        save_strategy="no",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    #treinar modelo
    trainer.train()

    #avaliar conjunto de teste
    results = trainer.evaluate()
    print(f"Acurácia: {results['eval_accuracy']}")
    print(f"F1-Score: {results['eval_f1']}")
    print(f"Precisão: {results['eval_precision']}")
    print(f"Recall: {results['eval_recall']}")

    #prever no conjunto de teste e plotar matriz
    predictions = trainer.predict(test_dataset)
    preds = predictions.predictions.argmax(-1)
    plot_confusion_matrix(test_df['labels'], preds, fold_number=fold_number + 1)

    #armazenar resultados para o plot final
    results_list.append((test_df['labels'], preds))

#plotar todas as matrizes de confusão combinadas
plot_combined_confusion_matrices(results_list)

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, KFold

#carregar e preparar df
df = balanced_df.copy()
df = df[['ementa', 'favoravel']]
label_mapping = {'banco': 0, 'cliente': 1}
df['labels'] = df['favoravel'].map(label_mapping)

#embaralhar rotulos (experimento de controle)
np.random.seed(42)
df['labels'] = np.random.permutation(df['labels'])

#dividir treino/validacao e teste
train_val_df, test_df = train_test_split(df, test_size=0.2, stratify=df['labels'], random_state=42)

#carregar modelo e tokenizador
model_name = "alfaneo/jurisbert-base-portuguese-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

#tokenizar textos
def tokenize(batch):
    return tokenizer(batch['ementa'], padding='max_length', truncation=True, max_length=512)

#calcular metricas
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

#plotar matriz de confusao
def plot_confusion_matrix(y_true, y_pred, fold_number=None, vmin=0, vmax=35):
    conf_matrix = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', xticklabels=['Banco', 'Cliente'], yticklabels=['Banco', 'Cliente'], vmin=vmin, vmax=vmax)
    plt.xlabel('Predito')
    plt.ylabel('Real')
    title = f'Confusao Fold {fold_number}' if fold_number else 'Confusao Final'
    plt.title(title)
    plt.show()

#plotar matrizes combinadas
def plot_combined_confusion_matrices(results_list):
    fig, axes = plt.subplots(1, len(results_list), figsize=(6 * len(results_list), 4))
    for i, (y_true, y_pred) in enumerate(results_list):
        conf_matrix = confusion_matrix(y_true, y_pred)
        ax = axes[i] if len(results_list) > 1 else axes
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='viridis', xticklabels=['Banco', 'Cliente'], yticklabels=['Banco', 'Cliente'], ax=ax)
        ax.set_title(f'Fold {i+1}')
    plt.suptitle('Confusao Folds')
    plt.show()

#preparar KFold
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results_list = []

#executar KFold
for fold_number, (train_index, val_index) in enumerate(kf.split(train_val_df)):
    train_df = train_val_df.iloc[train_index]
    val_df = train_val_df.iloc[val_index]

    #tokenizar datasets
    train_dataset = Dataset.from_pandas(train_df).map(tokenize, batched=True)
    val_dataset = Dataset.from_pandas(val_df).map(tokenize, batched=True)

    #carregar modelo
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    #configurar treinamento
    training_args = TrainingArguments(
        output_dir='/content/drive/MyDrive/TCC_FINAL/classified_data/treinamento',
        num_train_epochs=5,
        per_device_train_batch_size=8,
        evaluation_strategy="epoch",
        logging_dir='/content/drive/MyDrive/TCC_FINAL/classified_data/logs',
        learning_rate=5e-5,
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    #treinar modelo
    trainer.train()

    #avaliar validacao
    results = trainer.evaluate()
    print(f"\nFold {fold_number + 1} - Acuracia: {results['eval_accuracy']} F1: {results['eval_f1']} Prec: {results['eval_precision']} Recall: {results['eval_recall']}")

    #prever e plotar confusao
    predictions = trainer.predict(val_dataset)
    preds = predictions.predictions.argmax(-1)
    plot_confusion_matrix(val_df['labels'], preds, fold_number+1)
    results_list.append((val_df['labels'], preds))

#avaliar conjunto de teste final
test_dataset = Dataset.from_pandas(test_df).map(tokenize, batched=True)
test_trainer = Trainer(model=model, args=training_args, eval_dataset=test_dataset, compute_metrics=compute_metrics)
test_results = test_trainer.evaluate()
print(f"\nTeste Final - Acuracia: {test_results['eval_accuracy']} F1: {test_results['eval_f1']} Prec: {test_results['eval_precision']} Recall: {test_results['eval_recall']}")

#prever e plotar confusao final
test_predictions = test_trainer.predict(test_dataset)
test_preds = test_predictions.predictions.argmax(-1)
plot_confusion_matrix(test_df['labels'], test_preds)

#plotar todas as matrizes de confusao
plot_combined_confusion_matrices(results_list)

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings
import obonet
import os
warnings.filterwarnings('ignore')

from Dataset.GoDataset import GOTermDataset, get_class_frequencies_from_dataframe
from Model.ProtienGOClassifer import ProteinGOClassifier
from Dataset.utils import prepare_data, read_fasta


def validate(model, dataloader, device):
    """Validate the model - FIXED VERSION."""
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating', leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            
            predictions = (torch.sigmoid(logits) > 0.5).float()
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # FIXED: Convert to boolean
    all_predictions = np.vstack(all_predictions).astype(bool)
    all_labels = np.vstack(all_labels).astype(bool)

    hamming_acc = (all_predictions == all_labels).all(axis=1).mean()
    
    # FIXED: Use logical operators
    tp = np.logical_and(all_predictions, all_labels)
    fp = np.logical_and(all_predictions, np.logical_not(all_labels))
    fn = np.logical_and(np.logical_not(all_predictions), all_labels)

    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    
    return {
        'hamming_acc': hamming_acc,
        'precision': precision.mean(),
        'recall': recall.mean(),
        'f1': f1.mean()
    }

def test_go_classifier(
    train_term_df,
    train_seq,
    model,
    model_name,
    top_k=100,
    batch_size=16,
    max_length=512,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    alpha_method='effective_number',  # NEW PARAMETER
    focal_gamma=2.0,
    beta=0.999  # For effective_number method
):
    print("="*80)
    print("GO TERM PREDICTION WITH PER-CLASS FOCAL LOSS")
    print("="*80)

    # Prepare data
    print("\n[1/7] Preparing data...")
    data = prepare_data(train_term_df, train_seq, top_k=top_k, test_size=0.1)

    # NEW: Compute per-class alpha values
    print(f"\n[2/7] Computing per-class alpha using '{alpha_method}' method...")
    class_frequencies = get_class_frequencies_from_dataframe(
        train_term_df, data['top_terms']
    )

    print(f"Class frequency statistics:")
    print(f"  Min: {class_frequencies.min()}")
    print(f"  Max: {class_frequencies.max()}")
    print(f"  Mean: {class_frequencies.mean():.1f}")
    print(f"  Ratio (max/min): {class_frequencies.max() / class_frequencies.min():.1f}x")

    # Compute per-class alpha

    # Initialize tokenizer
    print(f"\n[3/7] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create datasets
    print("\n[4/7] Creating datasets...")

    val_dataset = GOTermDataset(
        data['val_sequences'],
        data['val_labels'],
        tokenizer,
        max_length=max_length
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    # Initialize model
    print(f"\n[5/7] Initializing model on {device}...")

    model = model.to(device)

    # Validate
    val_metrics = validate(model, val_loader, device)


    print(f"Val F1: {val_metrics['f1']:.4f}")
    print(f"Val Precision: {val_metrics['precision']:.4f}")
    print(f"Val Recall: {val_metrics['recall']:.4f}")

    return None

if __name__ == "__main__":

    
    BASE_PATH = "./cafa-6-protein-function-prediction/"

    
    go_graph = obonet.read_obo(os.path.join(BASE_PATH, 'Train/go-basic.obo'))
    print(f"Gene Ontology graph loaded with {len(go_graph)} nodes and {len(go_graph.edges)} edges.")

    train_terms_df = pd.read_csv(os.path.join(BASE_PATH, 'Train/train_terms.tsv'), sep='\\t')
    print(f"Training terms loaded. Shape: {train_terms_df.shape}")


    train_fasta_path = os.path.join(BASE_PATH, 'Train/train_sequences.fasta')

    test_fasta_path = os.path.join(BASE_PATH, 'Test/testsuperset.fasta')

    ia_df = pd.read_csv(os.path.join(BASE_PATH, 'IA.tsv'), sep='\\t', header=None, names=['term_id', 'ia_score'])
    ia_map = dict(zip(ia_df['term_id'], ia_df['ia_score']))
    print(train_terms_df['aspect'].value_counts().reset_index())

    train_seq = read_fasta(train_fasta_path)


    saved = torch.load('./saved/best_go_model_(256).pt', weights_only=False, map_location='cpu')
    model = ProteinGOClassifier(
        model_name='facebook/esm2_t6_8M_UR50D',
        num_classes=saved['model_state_dict']['classifier.3.weight'].shape[0],
        dropout=0.3,
        hidden_dim=512
    )
    model.load_state_dict(saved['model_state_dict'])

    test_go_classifier(
        train_term_df=train_terms_df,
        train_seq=train_seq,
        model=model,
        model_name='facebook/esm2_t6_8M_UR50D',
        top_k=256,
        batch_size=16,
        max_length=512,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        alpha_method='effective_number',
    )
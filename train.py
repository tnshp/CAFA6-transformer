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

from Utils.FocalLoss import PerClassFocalLoss, compute_per_class_alpha_from_frequencies
from Dataset.GoDataset import GOTermDataset, get_class_frequencies_from_dataframe
from Model.ProtienGOClassifer import ProteinGOClassifier
from Dataset.utils import prepare_data, read_fasta

def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0

    for batch in tqdm(dataloader, desc='Training'):
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        logits = model(input_ids, attention_mask)

        # Calculate loss
        loss = criterion(logits, labels)

        # Backward pass
        loss.backward()

        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def validate(model, dataloader, criterion,  device, ia_scores=None):
    """Validate the model - FIXED VERSION."""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validating', leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            predictions = (torch.sigmoid(logits) > 0.5).float()
            
            all_predictions.append(predictions.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    # FIXED: Convert to boolean
    all_predictions = np.vstack(all_predictions).astype(bool)
    all_labels = np.vstack(all_labels).astype(bool)
    
    hamming_acc = (all_predictions == all_labels).all(axis=1).mean()
        
    tp = np.logical_and(all_predictions, all_labels).sum(axis=0)
    fp = np.logical_and(all_predictions, np.logical_not(all_labels)).sum(axis=0)
    fn = np.logical_and(np.logical_not(all_predictions), all_labels).sum(axis=0)
    
    precision = (tp / (tp + fp + 1e-10))
    recall = (tp / (tp + fn + 1e-10))

    if ia_scores is not None:
        precision = (precision * ia_scores).mean()
        recall = (recall * ia_scores).mean()
    else:
        precision = precision.mean()
        recall = recall.mean()
    f1 = 2 * precision * recall / (precision + recall + 1e-10)
    
    # Log metrics
    
    # print(f"\nValidation Hamming Acc: {hamming_acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    return {
        'loss': total_loss / len(dataloader),
        'hamming_acc': hamming_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

print("✓ Fixed validate() function loaded!")


# ============================================================================
# 6. MAIN TRAINING PIPELINE
# ============================================================================
def train_go_classifier(
    train_term_df,
    train_seq,
    ia_df,
    model_name='facebook/esm2_t6_8M_UR50D',
    classifier_depth=1,
    dropout=0.3,
    hidden_dim=512,
    top_k=100,
    test_size=0.2,
    batch_size=16,
    num_epochs=10,
    learning_rate=2e-5,
    max_length=512,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    alpha_method='effective_number',  # NEW PARAMETER
    focal_gamma=2.0,
    beta=0.999,  # For effective_number method
    save_dir = None
):
    print("="*80)
    print("GO TERM PREDICTION WITH PER-CLASS FOCAL LOSS")
    print("="*80)

    # Prepare data
    print("\n[1/7] Preparing data...")
    data = prepare_data(train_term_df, train_seq, ia_df, top_k=top_k, test_size=test_size)

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
    per_class_alpha = compute_per_class_alpha_from_frequencies(
        class_frequencies, 
        method=alpha_method,
        beta=beta
    )

    print(f"\nPer-class alpha statistics:")
    print(f"  Min: {per_class_alpha.min():.4f}")
    print(f"  Max: {per_class_alpha.max():.4f}")
    print(f"  Mean: {per_class_alpha.mean():.4f}")
    print(f"  Ratio (max/min): {(per_class_alpha.max() / per_class_alpha.min()):.1f}x")

    # Initialize tokenizer
    print(f"\n[3/7] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Create datasets
    print("\n[4/7] Creating datasets...")
    train_dataset = GOTermDataset(
        data['train_sequences'],
        data['train_labels'],
        tokenizer,
        max_length=max_length
    )

    val_dataset = GOTermDataset(
        data['val_sequences'],
        data['val_labels'],
        tokenizer,
        max_length=max_length
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )

    # Initialize model
    print(f"\n[5/7] Initializing model on {device}...")
    model = ProteinGOClassifier(
        model_name=model_name,
        num_classes=data['num_classes'],
        classifier_depth=classifier_depth,
        dropout=dropout,
        hidden_dim=hidden_dim
    )
    model = model.to(device)

    # Initialize LOSS with per-class alpha
    criterion = PerClassFocalLoss(
        alpha=per_class_alpha,
        gamma=focal_gamma
    )

    print(f"\nUsing PerClassFocalLoss with:")
    print(f"  - Alpha: per-class (method={alpha_method})")
    print(f"  - Gamma: {focal_gamma}")

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )

    # Training loop
    print(f"\n[6/7] Training for {num_epochs} epochs...")
    print("-"*80)

    history = {
        'train_loss': [],
        'val_loss': [],
        'val_f1': [],
        'val_precision': [],
        'val_recall': []
    }

    best_f1 = 0.0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # Validate
        val_metrics = validate(model, val_loader, criterion, device, ia_scores=data['ia_scores'])

        # Update scheduler
        scheduler.step(val_metrics['loss'])

        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_metrics['loss'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])

        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val F1: {val_metrics['f1']:.4f}")
        print(f"Val Precision: {val_metrics['precision']:.4f}")
        print(f"Val Recall: {val_metrics['recall']:.4f}")

        # Save best model
        from datetime import datetime
        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            # save in date-stamped directory
            # create date-stamped directory if not exists
            if save_dir is None:
                save_dir = "./checkpoints"
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            os.makedirs(os.path.join(save_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")), exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'history': history,
                'optimizer_state_dict': optimizer.state_dict(),
                'val_f1': best_f1,
                'mlb': data['mlb'],
                'top_terms': data['top_terms'],
                'per_class_alpha': per_class_alpha,
                'alpha_method': alpha_method
            }, os.path.join(save_dir, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), 'best_go_model_focal.pt'))
            print(f"✓ Saved best model (F1: {best_f1:.4f})")

    print("\n[7/7] Training complete!")
    print("="*80)

    return model, tokenizer, history, data, per_class_alpha


# ============================================================================
# 7. INFERENCE FUNCTION
# ============================================================================

def predict_go_terms(model, sequence, tokenizer, mlb, device, threshold=0.5, max_length=512):
    """
    Predict GO terms for a single protein sequence.

    Args:
        model: Trained model
        sequence: Protein sequence string
        tokenizer: Tokenizer
        mlb: MultiLabelBinarizer
        device: Device
        threshold: Prediction threshold
        max_length: Max sequence length

    Returns:
        List of predicted GO terms with scores
    """
    model.eval()

    # Tokenize
    spaced_sequence = ' '.join(list(sequence))
    encoding = tokenizer(
        spaced_sequence,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    # Predict
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits).cpu().numpy()[0]

    # Get predictions
    predicted_indices = np.where(probs > threshold)[0]
    predicted_terms = mlb.classes_[predicted_indices]
    predicted_scores = probs[predicted_indices]

    # Sort by score
    sorted_idx = np.argsort(predicted_scores)[::-1]

    results = [
        {'term': predicted_terms[i], 'score': float(predicted_scores[i])}
        for i in sorted_idx
    ]

    return results



if __name__ == "__main__":

    import json
    configs = json.load(open('configs.json'))

    # Set paths
    data = configs.get('data_paths', {})
    BASE_PATH = data.get('base_path', "./cafa-6-protein-function-prediction/")

    go_graph = obonet.read_obo(os.path.join(BASE_PATH, 'Train/go-basic.obo'))
    print(f"Gene Ontology graph loaded with {len(go_graph)} nodes and {len(go_graph.edges)} edges.")
    # --- Load Training Terms ---
    train_terms_df = pd.read_csv(os.path.join(BASE_PATH, 'Train/train_terms.tsv'), sep='\\t')
    print(f"Training terms loaded. Shape: {train_terms_df.shape}")

    # --- Load Training Sequences ---
    # We will parse the FASTA file later when we need the sequences.
    train_fasta_path = os.path.join(BASE_PATH, 'Train/train_sequences.fasta')
    print(f"Training sequences path set: {train_fasta_path}")

    # --- Load Test Sequences ---
    test_fasta_path = os.path.join(BASE_PATH, 'Test/testsuperset.fasta')
    print(f"Test sequences path set: {test_fasta_path}")

    # --- Load Information Accretion (Weights) ---
    ia_df = pd.read_csv(os.path.join(BASE_PATH, 'IA.tsv'), sep='\\t', header=None, names=['term_id', 'ia_score'])
    ia_map = dict(zip(ia_df['term_id'], ia_df['ia_score']))
    print(f"Information Accretion scores loaded for {len(ia_map)} terms.")

    # --- Display a sample of the training terms data ---
    print("\\nSample of train_terms.tsv:")
    print(train_terms_df.head())

    # Table 5: Summary of GO Term Distribution in Training Data
    print("\\nTable 5: Summary of GO Term Distribution in Training Data")
    print(train_terms_df['aspect'].value_counts().reset_index())

    train_seq = read_fasta(train_fasta_path)
    
    model_configs = configs.get('model_configs', {})
    # Train the model
    model, tokenizer, history, data , alpha = train_go_classifier(
        train_term_df=train_terms_df,
        train_seq=train_seq,
        ia_df=ia_df,
        **model_configs
    )   


    # Make predictions
    test_sequence = "MKTIIALSYIFCLVFA..."
    predictions = predict_go_terms(
        model=model,
        sequence=test_sequence, 
        tokenizer=tokenizer,
        mlb=data['mlb'],
        device='cuda'
    )

    print("Predicted GO terms:")
    for pred in predictions:
        print(f"{pred['term']}: {pred['score']:.4f}")   
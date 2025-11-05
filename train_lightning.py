import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
import warnings
import obonet
import os
import json
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

warnings.filterwarnings('ignore')

from Utils.FocalLoss import compute_per_class_alpha_from_frequencies
from Dataset.GoDataset import GOTermDataset, get_class_frequencies_from_dataframe
from Model.ProtienGOClassifier_pl import ProteinGOClassifierLightning
from Dataset.utils import prepare_data, read_fasta

 
def train_go_classifier_lightning(
    train_term_df,
    train_seq,
    ia_df,
    model_name='facebook/esm2_t6_8M_UR50D',
    test_size=0.2, 
    top_k=100,
    batch_size=16,
    num_epochs=10,
    learning_rate=2e-5,
    max_length=512,
    alpha_method='effective_number',
    focal_gamma=2.0,
    beta=0.999,
    num_workers=2,
    accelerator='auto',
    devices='auto',
    precision='16-mixed',
    gradient_clip_val=1.0,
    accumulate_grad_batches=1,
    save_dir='./lightning_logs',
    checkpoint_dir='./checkpoints',
    run_name=None,
    classifier_depth=1,
    dropout=0.3,
    hidden_dim=512,
    save_top_k=3,
    patience=5,
):
    """
    Train GO classifier using PyTorch Lightning.
    
    Args:
        train_term_df: DataFrame with columns [EntryID, term, aspect]
        train_seq: Dictionary {EntryID: sequence}
        model_name: Pretrained transformer model name
        top_k: Number of most frequent GO terms to use
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        max_length: Maximum sequence length
        alpha_method: Method for computing per-class alpha
        focal_gamma: Gamma parameter for focal loss
        beta: Beta parameter for effective_number method
        num_workers: Number of dataloader workers
        accelerator: Accelerator type ('auto', 'gpu', 'cpu', 'tpu')
        devices: Number of devices or device IDs
        precision: Training precision ('32', '16-mixed', 'bf16-mixed')
        gradient_clip_val: Gradient clipping value
        accumulate_grad_batches: Number of batches to accumulate gradients
        save_dir: Directory for logging
        checkpoint_dir: Directory for model checkpoints
        classifier_depth: Number of hidden layers in MLP
        dropout: Dropout rate
        hidden_dim: Hidden dimension for MLP
        save_top_k: Number of best models to save
        patience: Patience for early stopping
    
    Returns:
        Trained model, tokenizer, trainer, data dictionary
    """
    print("="*80)
    print("GO TERM PREDICTION WITH PYTORCH LIGHTNING")
    print("="*80)
    
    if run_name is None:
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_name = f'run_{timestamp}'

    # Update directories to include run name
    run_save_dir = os.path.join(save_dir, run_name)
    run_checkpoint_dir = os.path.join(checkpoint_dir, run_name)

    os.makedirs(run_save_dir, exist_ok=True)
    os.makedirs(run_checkpoint_dir, exist_ok=True)

    print("="*80)
    print("GO TERM PREDICTION WITH PYTORCH LIGHTNING")
    print("="*80)
    print(f"\nRun name: {run_name}")
    print(f"Logs directory: {run_save_dir}")
    print(f"Checkpoints directory: {run_checkpoint_dir}")
    # Prepare data
    print("\n[1/6] Preparing data...")
    data = prepare_data(train_term_df, train_seq, ia_df, top_k=top_k, test_size=test_size)
    top_terms = pd.DataFrame(data['top_terms'], columns=['terms'])
    save_terms_path = os.path.join(run_checkpoint_dir, f'top_terms_{top_k}.csv')
    os.makedirs(run_save_dir, exist_ok=True)
    top_terms.to_csv(save_terms_path, index=False)
    print(f"Saved top terms to {save_terms_path}")
    
    # Compute per-class alpha values
    print(f"\n[2/6] Computing per-class alpha using '{alpha_method}' method...")
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
    print(f"\n[3/6] Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Create datasets
    print("\n[4/6] Creating datasets...")
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
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    # Initialize Lightning model
    print(f"\n[5/6] Initializing Lightning model...")

    model = ProteinGOClassifierLightning(
        model_name=model_name,
        num_classes=data['num_classes'],
        classifier_depth=classifier_depth,
        dropout=dropout,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        per_class_alpha=per_class_alpha,
        ia_scores=data['ia_scores'],
        focal_gamma=focal_gamma
    )
    
    print(f"\nModel configuration:")
    print(f"  - Model name: {model_name}")
    print(f"  - Number of classes: {data['num_classes']}")
    print(f"  - Classifier depth: {classifier_depth}")
    print(f"  - Hidden dim: {hidden_dim}")
    print(f"  - Dropout: {dropout}")
    print(f"  - Learning rate: {learning_rate}")
    print(f"  - Focal gamma: {focal_gamma}")
    print(f"  - Alpha method: {alpha_method}")
    
    # Setup callbacks
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=run_checkpoint_dir,
        filename='go-classifier-{epoch:02d}-{val_f1:.4f}',
        monitor='val_f1',
        mode='max',
        save_top_k=save_top_k,
        save_last=True,
        verbose=True
    )
    
    early_stop_callback = EarlyStopping(
        monitor='val_f1',
        patience=patience,
        mode='max',
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    
    # Setup logger
    logger = TensorBoardLogger(save_dir, name='go_classifier', version=run_name)
    
    # Initialize trainer
    print(f"\n[6/6] Initializing PyTorch Lightning Trainer...")
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator=accelerator,
        devices=devices,
        precision=precision,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulate_grad_batches,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
        enable_progress_bar=True,
        enable_model_summary=True,
        deterministic=False
    )
    
    print("\nTrainer configuration:")
    print(f"  - Max epochs: {num_epochs}")
    print(f"  - Accelerator: {accelerator}")
    print(f"  - Devices: {devices}")
    print(f"  - Precision: {precision}")
    print(f"  - Gradient clip: {gradient_clip_val}")
    print(f"  - Accumulate batches: {accumulate_grad_batches}")
    
    # Train the model
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80 + "\n")
    
    trainer.fit(model, train_loader, val_loader)
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nBest model saved at: {checkpoint_callback.best_model_path}")
    print(f"Best validation F1: {checkpoint_callback.best_model_score:.4f}")
    
    # Save additional metadata
    metadata = {
        'top_terms': data['top_terms'],
        'mlb': data['mlb'],
        'per_class_alpha': per_class_alpha.tolist(),
        'alpha_method': alpha_method,
        'model_name': model_name,
        'top_k': top_k,
        'max_length': max_length
    }
    
    metadata_path = os.path.join(checkpoint_dir, 'metadata.pt')
    torch.save(metadata, metadata_path)
    print(f"Metadata saved to: {metadata_path}")
    
    return model, tokenizer, trainer, data


def load_trained_model(checkpoint_path, metadata_path):
    """
    Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        metadata_path: Path to metadata file
    
    Returns:
        model, tokenizer, metadata
    """
    # Load metadata
    metadata = torch.load(metadata_path)
    
    # Load model
    model = ProteinGOClassifierLightning.load_from_checkpoint(checkpoint_path)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(metadata['model_name'])
    
    print(f"Model loaded from: {checkpoint_path}")
    print(f"Model name: {metadata['model_name']}")
    print(f"Number of classes: {len(metadata['top_terms'])}")
    
    return model, tokenizer, metadata
if __name__ == "__main__":
    import json
    configs = json.load(open('configs_pl.json'))

    # Set paths
    data = configs.get('data_paths', {})
    BASE_PATH = data.get('base_path', "./cafa-6-protein-function-prediction/")
    
    go_graph = obonet.read_obo(os.path.join(BASE_PATH, 'Train/go-basic.obo'))
    print(f"Gene Ontology graph loaded with {len(go_graph)} nodes and {len(go_graph.edges)} edges.")
    
    train_terms_df = pd.read_csv(os.path.join(BASE_PATH, 'Train/train_terms.tsv'), sep='\t')
    print(f"Training terms loaded. Shape: {train_terms_df.shape}")
    
    train_fasta_path = os.path.join(BASE_PATH, 'Train/train_sequences.fasta')
    print(f"Training sequences path set: {train_fasta_path}")
    
    test_fasta_path = os.path.join(BASE_PATH, 'Test/testsuperset.fasta')
    print(f"Test sequences path set: {test_fasta_path}")
    
    ia_df = pd.read_csv(os.path.join(BASE_PATH, 'IA.tsv'), sep='\t', header=None, names=['term_id', 'ia_score'])
    ia_map = dict(zip(ia_df['term_id'], ia_df['ia_score']))
    print(f"Information Accretion scores loaded for {len(ia_map)} terms.")
    
    # --- Display a sample of the training terms data ---
    print("\nSample of train_terms.tsv:")
    print(train_terms_df.head())
    
    # Table 5: Summary of GO Term Distribution in Training Data
    print("\nTable 5: Summary of GO Term Distribution in Training Data")
    print(train_terms_df['aspect'].value_counts().reset_index())
    
    train_seq = read_fasta(train_fasta_path)
    
    # Train the model
    # read json configs file
    
    
    # Pass train_term_df and train_seq along with other configs
    model_configs = configs.get('model_configs', {})

    model, tokenizer, history, data, _ = train_go_classifier_lightning(
        train_term_df=train_terms_df,
        train_seq=train_seq,
        ia_df=ia_df,
        **model_configs
    )
    
    # Make predictions on example
    test_sequence = list(train_seq.values())[0]  # Use first training sequence as example
    predictions = model.predict_go_terms(
        sequence=test_sequence,
        tokenizer=tokenizer,
        mlb=data['mlb'],
        threshold=0.5
    )
    
    print("\nTop 10 predicted GO terms:")
    for i, pred in enumerate(predictions[:10], 1):
        print(f"{i}. {pred['term']}: {pred['score']:.4f}")

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from typing import List, Dict, Tuple
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


def prepare_data(train_term_df, train_seq, ia_df, top_k=100, test_size=0.2, random_state=42):
    """
    Prepare data for training.

    Args:
        train_term_df: DataFrame with columns [EntryID, term, aspect]
        train_seq: Dictionary {EntryID: sequence}
        top_k: Number of most frequent GO terms to use
        test_size: Fraction for validation split
        random_state: Random seed

    Returns:
        Dictionary with prepared data
    """
    print(f"Preparing data with top {top_k} GO terms...")

    # Step 1: Select top K most frequent GO terms
    term_counts = train_term_df['term'].value_counts()
    top_terms = term_counts.head(top_k).index.tolist()

    print(f"Top {top_k} terms selected")
    print(f"Frequency range: {term_counts.iloc[0]} to {term_counts.iloc[top_k-1]}")

    # Step 2: Filter data to only include top terms
    filtered_df = train_term_df[train_term_df['term'].isin(top_terms)]

    # Step 3: Group by EntryID to create multi-label format
    entry_terms = filtered_df.groupby('EntryID')['term'].apply(list).to_dict()

    # Step 4: Filter entries that have sequences
    valid_entries = [entry for entry in entry_terms.keys() if entry in train_seq]

    print(f"Number of valid protein entries: {len(valid_entries)}")

    # Step 5: Create sequences and labels lists
    sequences = [train_seq[entry] for entry in valid_entries]
    labels_list = [entry_terms[entry] for entry in valid_entries]

    # Step 6: Multi-label binarization
    mlb = MultiLabelBinarizer(classes=top_terms)
    labels_binary = mlb.fit_transform(labels_list)

    print(f"Label matrix shape: {labels_binary.shape}")
    print(f"Average labels per protein: {labels_binary.sum(axis=1).mean():.2f}")
    print(f"Label distribution - min: {labels_binary.sum(axis=0).min()}, "
          f"max: {labels_binary.sum(axis=0).max()}")

    # Step 7: Train-validation split
    (train_sequences, val_sequences, 
     train_labels, val_labels) = train_test_split(
        sequences, labels_binary, 
        test_size=test_size, 
        random_state=random_state
    )   

    ia_df['term_id'] = pd.Categorical(ia_df['term_id'], categories=top_terms, ordered=True)
    ia_df.sort_values('term_id')[0:len(top_terms)]
    ia_scores = ia_df.sort_values('term_id')[0:len(top_terms)]['ia_score'].values

    print(f"Train samples: {len(train_sequences)}")
    print(f"Validation samples: {len(val_sequences)}")

    return {
        'train_sequences': train_sequences,
        'val_sequences': val_sequences,
        'train_labels': train_labels,
        'val_labels': val_labels,
        'mlb': mlb,
        'top_terms': top_terms,
        'ia_scores': ia_scores,
        'num_classes': len(top_terms)
    }


# ------------------------------------------------------------
# tiny utils for FASTA/TSV/OBO parsing 
# ------------------------------------------------------------
def read_fasta(path: str) -> Dict[str, str]:
    seqs = {}
    with open(path, "r") as f:
        pid = None; seq_parts = []
        for line in f:
            line=line.strip()
            if line.startswith(">"):
                if pid: seqs[pid] = "".join(seq_parts)
                header=line[1:].split()[0]
                if "|" in header:
                    parts=header.split("|"); pid = parts[1] if len(parts)>=2 else header
                else:
                    pid = header
                seq_parts=[]
            else:
                seq_parts.append(line.strip())
        if pid: seqs[pid] = "".join(seq_parts)
    print(f"[io] Read {len(seqs)} sequences from {path}")
    return seqs
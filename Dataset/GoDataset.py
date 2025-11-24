
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np



def get_class_frequencies_from_dataframe(train_term_df, top_terms):
    """
    Compute class frequencies from the training data.

    Args:
        train_term_df: DataFrame with columns [EntryID, term, aspect]
        top_terms: List of GO term IDs (top K most frequent)

    Returns:
        class_frequencies: Array of shape (K,) with counts
    """
    from collections import Counter

    # Count frequency of each term
    term_counts = Counter(train_term_df['term'])

    # Get frequencies for top_terms in order
    class_frequencies = np.array([term_counts[term] for term in top_terms])

    return class_frequencies


class GOTermDataset(Dataset):
    """
    Dataset for GO term prediction from protein sequences.

    Args:
        sequences: List of protein sequences
        labels: Multi-hot encoded labels (num_samples, num_classes)
        tokenizer: Hugging Face tokenizer
        max_length: Maximum sequence length for tokenization
        oversample_indices: List of indices for oversampling (optional)
    """
    def __init__(self, sequences, labels, tokenizer, max_length=512, oversample_indices=None):
        self.sequences = sequences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.oversample_indices = oversample_indices if oversample_indices is not None else list(range(len(sequences)))

    def __len__(self):
        return len(self.oversample_indices)

    def __getitem__(self, idx):
        # Map idx to the actual sample index using oversample_indices
        sample_idx = self.oversample_indices[idx]
        sequence = self.sequences[sample_idx]
        label = self.labels[sample_idx]

        # Tokenize protein sequence
        spaced_sequence = ' '.join(list(sequence))

        encoding = self.tokenizer(
            spaced_sequence,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.FloatTensor(label)
        }

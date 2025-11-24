import numpy as np 
from Dataset.GoDataset import get_class_frequencies_from_dataframe


def resample(data, train_terms_df, strategy='inv_freq', I=1000, t=0.5):
    class_frequencies = get_class_frequencies_from_dataframe(train_terms_df, data['top_terms'])
    sampled_idx = []

    class_weights = 1.0 / (class_frequencies + 1e-6)
    labels = data['train_labels']
    top_terms_len = len(data['top_terms'])
    N = len(labels)
    ep = 1e-3

    if strategy == 'inv_freq':
        R = (labels * class_weights).sum(axis=1)

        for _ in range(I):
            k = np.random.randint(top_terms_len)
            class_idx = np.where(labels[:, k] == 1)[0]
            r = R[class_idx]
            r = r / r.sum()
            idx = np.random.choice(class_idx, p=r)
            sampled_idx.append(idx)
        return sampled_idx

    elif strategy == 'distribution_balanced':
        p_i = (labels * class_weights).sum(axis=1)

        for _ in range(I):
            k = np.random.randint(top_terms_len)
            p_c = class_weights[k]
            class_idx = np.where(labels[:, k] == 1)[0]
            r = p_c / p_i[class_idx]
            r = r / r.sum()
            idx = np.random.choice(class_idx, p=r)
            sampled_idx.append(idx)
        return sampled_idx

    elif strategy == 'log_pos':
        class_weights = np.log(t * N / (class_frequencies + 1e-6))
        R = (labels * class_weights).sum(axis=1)

        for _ in range(I):
            k = np.random.randint(top_terms_len)
            class_idx = np.where(labels[:, k] == 1)[0]
            r = R[class_idx]
            r = r - r.min() + ep
            r = r / r.sum()
            idx = np.random.choice(class_idx, p=r)
            sampled_idx.append(idx)
        return sampled_idx
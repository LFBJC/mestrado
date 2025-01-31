import numpy as np


def hv_block_cv_three_sets(data, n_splits, h, v):
    """
    Divide os dados em treino, validação e teste usando HV-Block Cross-Validation.

    Parâmetros:
    - data: array ou DataFrame contendo a série temporal.
    - n_splits: número de divisões.
    - h: tamanho dos conjuntos de validação e teste.
    - v: tamanho da zona de exclusão entre treino e validação, e entre validação e teste.

    Retorna:
    - Lista de tuplas (train_idx, val_idx, test_idx)
    """

    n = len(data)
    split_size = (n - (2 * h + 2 * v) * n_splits) // n_splits  # Tamanho do conjunto de treino

    indices = np.arange(n)
    splits = []

    for i in range(n_splits):
        # Definir os índices do conjunto de teste
        start_test = i * (split_size + 2 * h + 2 * v)
        end_test = start_test + h
        test_idx = indices[start_test:end_test]

        # Definir os índices do conjunto de validação (após a zona de exclusão v)
        start_val = end_test + v
        end_val = start_val + h
        val_idx = indices[start_val:end_val]

        # O conjunto de treino é tudo antes da validação (considerando buffer v antes da validação)
        train_idx = indices[:start_val - v]

        splits.append((train_idx, val_idx, test_idx))

    return splits


# Exemplo de uso:
data = np.arange(1000)  # Simulando uma série temporal
splits = hv_block_cv_three_sets(data, n_splits=5, h=50, v=30)

for i, (train_idx, val_idx, test_idx) in enumerate(splits):
    print(f"Split {i + 1}:")
    print(f"  Train: {train_idx[:5]}... ({len(train_idx)} samples)")
    print(f"  Validation: {val_idx[:5]}... ({len(val_idx)} samples)")
    print(f"  Test: {test_idx[:5]}... ({len(test_idx)} samples)\n")

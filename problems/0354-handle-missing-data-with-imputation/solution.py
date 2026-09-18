import numpy as np

def impute_missing_data(data: np.ndarray, strategy: str = 'mean') -> np.ndarray:
    """
    Impute missing values in a 2D array using the specified strategy.

    Args:
        data: 2D numpy array with missing values represented as np.nan
        strategy: Imputation strategy - 'mean', 'median', or 'mode'

    Returns:
        2D numpy array with missing values imputed
    """
    result = data.copy()

    if strategy not in ('mean', 'median', 'mode'):
        raise ValueError("strategy must be 'mean', 'median', or 'mode'")

    for col in range(result.shape[1]):
        column = result[:, col]

        # 只取非 NaN 值
        valid_values = column[~np.isnan(column)]

        # 整欄都是 NaN → 保持不變
        if valid_values.size == 0:
            continue

        if strategy == 'mean':
            fill_value = np.mean(valid_values)

        elif strategy == 'median':
            fill_value = np.median(valid_values)

        elif strategy == 'mode':
            values, counts = np.unique(valid_values, return_counts=True)
            fill_value = values[np.argmax(counts)]

        # 將該欄所有 NaN 替換掉
        column[np.isnan(column)] = fill_value

    return result
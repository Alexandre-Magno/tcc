import pandas as pd


def split_temporal_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    date_column: str = "data",
    target_column: str = "valor",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Realiza o split dos dados considerando a temporalidade.

    Args:
        df: DataFrame com os dados
        test_size: Proporção dos dados que será usada para teste (default: 0.2)
        date_column: Nome da coluna que contém as datas (default: 'data')

    Returns:
        X_train, X_test, y_train, y_test: Dados separados em treino e teste
    """
    # Ordenando os dados por data
    df = df.sort_values(date_column)

    # Calculando o índice de corte
    cut_idx = int(len(df) * (1 - test_size))

    # Separando os dados
    train_data = df.iloc[:cut_idx]
    test_data = df.iloc[cut_idx:]

    # Separando features e target
    X_train = train_data.drop(target_column, axis=1)
    y_train = train_data[target_column]

    X_test = test_data.drop(target_column, axis=1)
    y_test = test_data[target_column]

    return X_train, X_test, y_train, y_test

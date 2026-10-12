"""Prepara una copia de datos limpios para exploración, sin alterar el original."""
from churn.config import PROCESSED_DATA, RAW_DATA
from churn.dataset import clean_dataset, read_dataset


def main():
    clean = clean_dataset(read_dataset(RAW_DATA))
    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(PROCESSED_DATA, index=False)
    print(f"Guardado {PROCESSED_DATA}: {len(clean)} clientes")


if __name__ == "__main__":
    main()

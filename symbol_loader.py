import io
import requests
import pandas as pd


NSE_SYMBOL_MASTER_URL = (
    "https://public.fyers.in/sym_details/NSE_CM.csv"
)


def load_nse_equity_symbols():
    """
    Download the FYERS NSE symbol master and return
    NSE equity symbols in FYERS format.
    """

    print("Downloading FYERS NSE symbol master...")

    response = requests.get(
        NSE_SYMBOL_MASTER_URL,
        timeout=30
    )

    response.raise_for_status()

    df = pd.read_csv(
        io.BytesIO(response.content),
        header=None,
        low_memory=False
    )

    print(f"Downloaded {len(df):,} NSE records.")

    # Column 9 contains the FYERS symbol.
    # Column 16 contains the series.
    equity_df = df[
        (df[9].notna()) &
        (df[9].str.endswith("-EQ", na=False))
    ].copy()

    symbols = (
        equity_df[9]
        .drop_duplicates()
        .sort_values()
        .tolist()
    )

    print(f"Found {len(symbols):,} NSE equity symbols.")

    return symbols


if __name__ == "__main__":

    symbols = load_nse_equity_symbols()

    print("\nFirst 20 NSE equity symbols:")

    for symbol in symbols[:20]:
        print(symbol)

    print("\nTotal equity symbols:", len(symbols))
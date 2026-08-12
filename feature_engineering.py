import time
import pandas as pd
import numpy as np

from historical_data import (
    create_fyers_client,
    get_historical_data
)

from smma_engine import detect_crossovers


# ============================================================
# CONFIGURATION
# ============================================================

# Start with a small group of stocks for testing.
# Once this works, we will expand the dataset.
SYMBOLS = [
    "NSE:SBIN-EQ",
    "NSE:RELIANCE-EQ",
    "NSE:INFY-EQ",
    "NSE:TCS-EQ",
    "NSE:ICICIBANK-EQ",
    "NSE:HDFCBANK-EQ",
    "NSE:ITC-EQ",
    "NSE:AXISBANK-EQ",
    "NSE:LT-EQ",
    "NSE:BHARTIARTL-EQ"
]

HISTORICAL_DAYS = 60

OUTPUT_FILE = "ml_dataset.csv"


# ============================================================
# FEATURE CALCULATION
# ============================================================

def create_features(df):
    """
    Create machine-learning features from historical OHLCV data.

    IMPORTANT:
    All input features are based only on information available
    at the crossover candle.
    """

    df = df.copy()

    # --------------------------------------------------------
    # SMMA
    # --------------------------------------------------------

    result = detect_crossovers(df)

    # --------------------------------------------------------
    # Price-based features
    # --------------------------------------------------------

    result["return_1"] = (
        result["close"].pct_change(1) * 100
    )

    result["return_3"] = (
        result["close"].pct_change(3) * 100
    )

    result["return_6"] = (
        result["close"].pct_change(6) * 100
    )

    result["return_12"] = (
        result["close"].pct_change(12) * 100
    )

    # --------------------------------------------------------
    # Distance from SMMAs
    # --------------------------------------------------------

    result["smma_difference"] = (
        result["SMMA20"] - result["SMMA120"]
    )

    result["smma_difference_percent"] = (
        result["smma_difference"]
        / result["SMMA120"]
        * 100
    )

    result["price_vs_smma20_percent"] = (
        (
            result["close"]
            - result["SMMA20"]
        )
        / result["SMMA20"]
        * 100
    )

    result["price_vs_smma120_percent"] = (
        (
            result["close"]
            - result["SMMA120"]
        )
        / result["SMMA120"]
        * 100
    )

    # --------------------------------------------------------
    # Candle characteristics
    # --------------------------------------------------------

    result["candle_range_percent"] = (
        (
            result["high"]
            - result["low"]
        )
        / result["close"]
        * 100
    )

    result["body_percent"] = (
        (
            result["close"]
            - result["open"]
        )
        / result["open"]
        * 100
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    result["volatility_12"] = (
        result["return_1"]
        .rolling(12)
        .std()
    )

    result["volatility_24"] = (
        result["return_1"]
        .rolling(24)
        .std()
    )

    # --------------------------------------------------------
    # Volume features
    # --------------------------------------------------------

    result["volume_ma_12"] = (
        result["volume"]
        .rolling(12)
        .mean()
    )

    result["volume_ma_24"] = (
        result["volume"]
        .rolling(24)
        .mean()
    )

    result["volume_ratio"] = (
        result["volume"]
        / result["volume_ma_24"]
    )

    # --------------------------------------------------------
    # Trend strength
    # --------------------------------------------------------

    result["smma20_slope"] = (
        result["SMMA20"]
        .diff(3)
    )

    result["smma120_slope"] = (
        result["SMMA120"]
        .diff(3)
    )

    # --------------------------------------------------------
    # Time features
    # --------------------------------------------------------

    result["hour"] = (
        result["datetime"].dt.hour
    )

    result["minute"] = (
        result["datetime"].dt.minute
    )

    # --------------------------------------------------------
    # Keep only crossover rows
    # --------------------------------------------------------

    result = result[
        result["signal"] != ""
    ].copy()

    return result


# ============================================================
# CREATE TRADE LABELS
# ============================================================

def create_trade_labels(crossover_df):
    """
    For every crossover:

    BUY:
        profit = exit_price - entry_price

    SELL:
        profit = entry_price - exit_price

    The next crossover is used as the exit point.

    The final crossover is ignored because it has no
    completed trade.
    """

    df = crossover_df.copy()

    if len(df) < 2:
        return pd.DataFrame()

    trades = []

    for i in range(len(df) - 1):

        entry = df.iloc[i]
        exit_trade = df.iloc[i + 1]

        entry_price = float(
            entry["close"]
        )

        exit_price = float(
            exit_trade["close"]
        )

        signal = entry["signal"]

        # ----------------------------------------------------
        # Calculate profit/loss
        # ----------------------------------------------------

        if signal == "BUY":

            profit_loss = (
                exit_price
                - entry_price
            )

        elif signal == "SELL":

            profit_loss = (
                entry_price
                - exit_price
            )

        else:
            continue

        profitable = (
            1 if profit_loss > 0 else 0
        )

        trade_duration_minutes = (
            (
                exit_trade["datetime"]
                - entry["datetime"]
            ).total_seconds()
            / 60
        )

        row = entry.copy()

        row["exit_time"] = (
            exit_trade["datetime"]
        )

        row["exit_price"] = (
            exit_price
        )

        row["profit_loss"] = (
            profit_loss
        )

        row["profitable"] = (
            profitable
        )

        row["trade_duration_minutes"] = (
            trade_duration_minutes
        )

        trades.append(row)

    if not trades:
        return pd.DataFrame()

    return pd.DataFrame(trades)


# ============================================================
# PROCESS ONE STOCK
# ============================================================

def process_symbol(
    fyers,
    symbol,
    days=60
):

    print("\n" + "-" * 60)
    print(f"Processing {symbol}")
    print("-" * 60)

    try:

        df = get_historical_data(
            fyers,
            symbol,
            resolution="5",
            days=days
        )

        if df is None or df.empty:

            print(
                "No historical data received."
            )

            return None

        print(
            f"Received {len(df)} candles."
        )

        # ----------------------------------------------------
        # Create features
        # ----------------------------------------------------

        features = create_features(df)

        if features.empty:

            print(
                "No crossover signals found."
            )

            return None

        print(
            f"Detected {len(features)} "
            f"crossover signals."
        )

        # ----------------------------------------------------
        # Create labels
        # ----------------------------------------------------

        trades = create_trade_labels(
            features
        )

        if trades.empty:

            print(
                "Not enough crossovers "
                "to create completed trades."
            )

            return None

        # Add symbol

        trades["symbol"] = symbol

        # ----------------------------------------------------
        # Print summary
        # ----------------------------------------------------

        wins = int(
            trades["profitable"].sum()
        )

        total = len(trades)

        win_rate = (
            wins / total * 100
        )

        print(
            f"Completed trades: {total}"
        )

        print(
            f"Profitable trades: {wins}"
        )

        print(
            f"Win rate: {win_rate:.2f}%"
        )

        return trades

    except Exception as e:

        print(
            f"ERROR processing {symbol}:"
        )

        print(e)

        return None


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AI/ML STOCK SCREENER - FEATURE ENGINEERING")
    print("=" * 70)

    print(
        f"\nStocks to process: "
        f"{len(SYMBOLS)}"
    )

    print(
        f"Historical period: "
        f"{HISTORICAL_DAYS} days"
    )

    # --------------------------------------------------------
    # Connect to FYERS
    # --------------------------------------------------------

    print("\nConnecting to FYERS...")

    fyers = create_fyers_client()

    print(
        "FYERS connection successful."
    )

    # --------------------------------------------------------
    # Process stocks
    # --------------------------------------------------------

    all_datasets = []

    for index, symbol in enumerate(
        SYMBOLS,
        start=1
    ):

        print(
            f"\n[{index}/{len(SYMBOLS)}]"
        )

        result = process_symbol(
            fyers,
            symbol,
            days=HISTORICAL_DAYS
        )

        if (
            result is not None
            and not result.empty
        ):

            all_datasets.append(
                result
            )

        # Small delay to avoid
        # hitting API too aggressively

        time.sleep(0.5)

    # --------------------------------------------------------
    # Combine datasets
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("COMBINING DATASETS")
    print("=" * 70)

    if not all_datasets:

        print(
            "\nNo datasets were created."
        )

        return

    dataset = pd.concat(
        all_datasets,
        ignore_index=True
    )

    # --------------------------------------------------------
    # Remove rows with missing values
    # --------------------------------------------------------

    print(
        f"\nRaw ML rows: "
        f"{len(dataset)}"
    )

    # Columns required for ML

    feature_columns = [
        "signal",
        "close",
        "SMMA20",
        "SMMA120",
        "smma_difference",
        "smma_difference_percent",
        "price_vs_smma20_percent",
        "price_vs_smma120_percent",
        "return_1",
        "return_3",
        "return_6",
        "return_12",
        "candle_range_percent",
        "body_percent",
        "volatility_12",
        "volatility_24",
        "volume",
        "volume_ma_12",
        "volume_ma_24",
        "volume_ratio",
        "smma20_slope",
        "smma120_slope",
        "hour",
        "minute",
        "profitable",
        "profit_loss"
    ]

    dataset = dataset.dropna(
        subset=feature_columns
    ).copy()

    # --------------------------------------------------------
    # Reorder columns
    # --------------------------------------------------------

    columns = [
        "symbol",
        "datetime",
        "signal",
        "close",
        "SMMA20",
        "SMMA120",
        "smma_difference",
        "smma_difference_percent",
        "price_vs_smma20_percent",
        "price_vs_smma120_percent",
        "return_1",
        "return_3",
        "return_6",
        "return_12",
        "candle_range_percent",
        "body_percent",
        "volatility_12",
        "volatility_24",
        "volume",
        "volume_ma_12",
        "volume_ma_24",
        "volume_ratio",
        "smma20_slope",
        "smma120_slope",
        "hour",
        "minute",
        "exit_time",
        "exit_price",
        "profit_loss",
        "profitable",
        "trade_duration_minutes"
    ]

    dataset = dataset[
        [
            col
            for col in columns
            if col in dataset.columns
        ]
    ]

    # --------------------------------------------------------
    # Save dataset
    # --------------------------------------------------------

    dataset.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------------
    # Final statistics
    # --------------------------------------------------------

    print(
        f"\nFinal ML dataset rows: "
        f"{len(dataset)}"
    )

    print(
        f"Stocks represented: "
        f"{dataset['symbol'].nunique()}"
    )

    print(
        f"BUY signals: "
        f"{(dataset['signal'] == 'BUY').sum()}"
    )

    print(
        f"SELL signals: "
        f"{(dataset['signal'] == 'SELL').sum()}"
    )

    profitable_count = int(
        dataset["profitable"].sum()
    )

    losing_count = (
        len(dataset)
        - profitable_count
    )

    print(
        f"Profitable trades: "
        f"{profitable_count}"
    )

    print(
        f"Losing trades: "
        f"{losing_count}"
    )

    if len(dataset) > 0:

        overall_win_rate = (
            profitable_count
            / len(dataset)
            * 100
        )

        print(
            f"Overall win rate: "
            f"{overall_win_rate:.2f}%"
        )

    print(
        f"\nDataset saved to:"
        f"\n{OUTPUT_FILE}"
    )

    print("\nFirst 10 rows:")

    print(
        dataset.head(10).to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
import os
from datetime import datetime, timedelta

import pandas as pd
from dotenv import load_dotenv
from fyers_apiv3 import fyersModel


def create_fyers_client():

    load_dotenv()

    client_id = os.getenv("FYERS_CLIENT_ID")

    if not client_id:
        raise ValueError(
            "FYERS_CLIENT_ID not found in .env"
        )

    if not os.path.exists("access_token.txt"):
        raise FileNotFoundError(
            "access_token.txt not found."
        )

    with open("access_token.txt", "r") as file:
        access_token = file.read().strip()

    if not access_token:
        raise ValueError(
            "Access token is empty."
        )

    return fyersModel.FyersModel(
        client_id=client_id,
        token=access_token,
        is_async=False,
        log_path=""
    )


def get_historical_data(
    fyers,
    symbol,
    resolution="5",
    days=30
):

    end_date = datetime.now()
    start_date = end_date - timedelta(
        days=days
    )

    data = {
        "symbol": symbol,
        "resolution": resolution,
        "date_format": "1",
        "range_from": start_date.strftime(
            "%Y-%m-%d"
        ),
        "range_to": end_date.strftime(
            "%Y-%m-%d"
        ),
        "cont_flag": "1"
    }

    response = fyers.history(
        data=data
    )

    if response.get("s") != "ok":

        print("History API error:")
        print(response)

        return None

    candles = response.get(
        "candles",
        []
    )

    if not candles:
        return None

    df = pd.DataFrame(
        candles,
        columns=[
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    )

    df["datetime"] = pd.to_datetime(
        df["timestamp"],
        unit="s"
    )

    return df[
        [
            "datetime",
            "open",
            "high",
            "low",
            "close",
            "volume"
        ]
    ]


if __name__ == "__main__":

    print("=" * 60)
    print("FYERS HISTORICAL DATA TEST")
    print("=" * 60)

    fyers = create_fyers_client()

    symbol = "NSE:SBIN-EQ"

    print(
        f"\nDownloading historical data "
        f"for {symbol}..."
    )

    df = get_historical_data(
        fyers,
        symbol,
        resolution="5",
        days=30
    )

    if df is None:

        print(
            "\nNo historical data received."
        )

    else:

        print(
            f"\nReceived {len(df)} candles."
        )

        print("\nLast 10 candles:")

        print(
            df.tail(10).to_string(
                index=False
            )
        )
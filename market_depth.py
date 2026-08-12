import os

from dotenv import load_dotenv
from fyers_apiv3 import fyersModel


MIN_LIQUIDITY = 1_000_000


def create_fyers_client():

    load_dotenv()

    client_id = os.getenv("FYERS_CLIENT_ID")

    if not client_id:
        raise ValueError(
            "FYERS_CLIENT_ID not found in .env"
        )

    if not os.path.exists("access_token.txt"):
        raise FileNotFoundError(
            "access_token.txt not found. Run auth.py first."
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


def get_market_depth(fyers, symbol):

    data = {
        "symbol": symbol,
        "ohlcv_flag": 1
    }

    response = fyers.depth(data=data)

    if response.get("s") != "ok":
        return None

    return response.get("d", {}).get(symbol)


def extract_depth_data(depth):

    if not depth:
        return None

    bids = depth.get("bids", [])
    asks = depth.get("ask", [])

    # Best bid
    best_bid = bids[0] if bids else {}

    # Best ask
    best_ask = asks[0] if asks else {}

    total_bid_quantity = depth.get(
        "totalbuyqty",
        0
    )

    total_ask_quantity = depth.get(
        "totalsellqty",
        0
    )

    result = {

        "ltp": depth.get("ltp"),

        "ltq": depth.get("ltq"),

        "volume": depth.get("v"),

        "atp": depth.get("atp"),

        "bid_price": best_bid.get(
            "price",
            0
        ),

        "bid_quantity": best_bid.get(
            "volume",
            0
        ),

        "ask_price": best_ask.get(
            "price",
            0
        ),

        "ask_quantity": best_ask.get(
            "volume",
            0
        ),

        "total_bid_quantity": total_bid_quantity,

        "total_ask_quantity": total_ask_quantity,

        "liquidity_pass": (
            total_bid_quantity > MIN_LIQUIDITY
            and
            total_ask_quantity > MIN_LIQUIDITY
        )
    }

    return result


if __name__ == "__main__":

    print("=" * 60)
    print("FYERS MARKET DEPTH TEST")
    print("=" * 60)

    fyers = create_fyers_client()

    symbol = "NSE:SBIN-EQ"

    print(f"\nRequesting depth for {symbol}...")

    depth = get_market_depth(
        fyers,
        symbol
    )

    if depth is None:

        print("\nFailed to retrieve market depth.")

    else:

        result = extract_depth_data(
            depth
        )

        print("\nMarket Data")
        print("-" * 40)

        for key, value in result.items():

            print(
                f"{key:25}: {value}"
            )
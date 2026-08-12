import os
import time

import pandas as pd
from dotenv import load_dotenv
from fyers_apiv3 import fyersModel

from symbol_loader import load_nse_equity_symbols


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 50

MIN_LTP = 30
MAX_LTP = 500

# Minimum traded volume used as the primary
# liquidity condition.
MIN_VOLUME = 1_000_000

# Optional order-book liquidity threshold.
# This is NOT used as a rejection condition.
MIN_DEPTH_QUANTITY = 1_000_000

# Time delay between API requests.
QUOTE_DELAY = 0.2
DEPTH_DELAY = 0.1

# Output file
OUTPUT_FILE = "market_data.csv"


# ============================================================
# FYERS CLIENT
# ============================================================

def create_fyers_client():
    """
    Create and return an authenticated FYERS client.
    """

    load_dotenv()

    client_id = os.getenv("FYERS_CLIENT_ID")

    if not client_id:
        raise ValueError(
            "FYERS_CLIENT_ID not found in .env"
        )

    if not os.path.exists("access_token.txt"):
        raise FileNotFoundError(
            "access_token.txt not found. "
            "Run auth.py first."
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


# ============================================================
# CREATE BATCHES
# ============================================================

def create_batches(
    symbols,
    batch_size=BATCH_SIZE
):
    """
    Split symbols into smaller batches.
    """

    for i in range(
        0,
        len(symbols),
        batch_size
    ):
        yield symbols[
            i:i + batch_size
        ]


# ============================================================
# GET LIVE QUOTES
# ============================================================

def get_quotes(
    fyers,
    symbols
):
    """
    Download live quotes for all supplied symbols.

    FYERS allows multiple symbols in one request,
    so symbols are processed in batches.
    """

    all_quotes = []

    batches = list(
        create_batches(symbols)
    )

    print(
        f"\nRequesting quotes for "
        f"{len(symbols):,} symbols "
        f"in {len(batches)} batches..."
    )

    for index, batch in enumerate(
        batches,
        start=1
    ):

        symbol_string = ",".join(batch)

        try:

            response = fyers.quotes(
                data={
                    "symbols": symbol_string
                }
            )

            if response.get("s") != "ok":

                print(
                    f"Batch {index}/"
                    f"{len(batches)} failed:"
                )

                print(response)

                continue

            for item in response.get(
                "d",
                []
            ):

                if item.get("s") != "ok":
                    continue

                quote = item.get(
                    "v",
                    {}
                )

                all_quotes.append({

                    "symbol": item.get(
                        "n"
                    ),

                    "ltp": quote.get(
                        "lp"
                    ),

                    "bid_price": quote.get(
                        "bid"
                    ),

                    "ask_price": quote.get(
                        "ask"
                    ),

                    "volume": quote.get(
                        "volume"
                    ),

                    "open": quote.get(
                        "open_price"
                    ),

                    "high": quote.get(
                        "high_price"
                    ),

                    "low": quote.get(
                        "low_price"
                    ),

                    "previous_close":
                        quote.get(
                            "prev_close_price"
                        ),

                    "change":
                        quote.get(
                            "ch"
                        ),

                    "change_percent":
                        quote.get(
                            "chp"
                        ),

                    "average_traded_price":
                        quote.get(
                            "atp"
                        )
                })

            print(
                f"Batch {index}/"
                f"{len(batches)} completed "
                f"({len(batch)} symbols)"
            )

        except Exception as e:

            print(
                f"Error in batch "
                f"{index}/{len(batches)}: {e}"
            )

        time.sleep(
            QUOTE_DELAY
        )

    return pd.DataFrame(
        all_quotes
    )


# ============================================================
# PRICE FILTER
# ============================================================

def filter_by_ltp(df):
    """
    Keep stocks whose latest traded price
    is between MIN_LTP and MAX_LTP.
    """

    if df.empty:
        return df

    filtered = df[
        (df["ltp"] >= MIN_LTP)
        &
        (df["ltp"] <= MAX_LTP)
    ].copy()

    return filtered.reset_index(
        drop=True
    )


# ============================================================
# MARKET DEPTH
# ============================================================

def get_market_depth(
    fyers,
    symbol
):
    """
    Request live market depth for a symbol.

    FYERS may return zero for one side of the
    order book. This is treated as valid depth
    information rather than automatically rejecting
    the stock.
    """

    try:

        response = fyers.depth(
            data={
                "symbol": symbol,
                "ohlcv_flag": 1
            }
        )

        if response.get("s") != "ok":
            return None

        depth = response.get(
            "d",
            {}
        ).get(symbol)

        return depth

    except Exception as e:

        print(
            f"\nDepth error for "
            f"{symbol}: {e}"
        )

        return None


# ============================================================
# SAFE FLOAT CONVERSION
# ============================================================

def safe_float(value, default=0.0):
    """
    Safely convert a value to float.
    """

    try:

        if value is None:
            return default

        return float(value)

    except (
        TypeError,
        ValueError
    ):

        return default


# ============================================================
# EXTRACT MARKET DEPTH
# ============================================================

def extract_depth(depth):
    """
    Extract useful information from FYERS depth response.

    Important:
    FYERS can return zero quantities/prices on one side
    of the order book.

    Therefore:

        bid = 0
        ask > 0

    OR:

        bid > 0
        ask = 0

    is considered valid depth.

    Spread is calculated only when both bid and ask
    prices are valid.
    """

    if not depth:
        return None

    # --------------------------------------------------------
    # Order book
    # --------------------------------------------------------

    bids = depth.get(
        "bids",
        []
    )

    asks = depth.get(
        "ask",
        []
    )

    # --------------------------------------------------------
    # Best bid
    # --------------------------------------------------------

    best_bid = {}

    for bid in bids:

        price = safe_float(
            bid.get("price"),
            0
        )

        volume = safe_float(
            bid.get("volume"),
            0
        )

        if price > 0 and volume > 0:

            best_bid = bid
            break

    # --------------------------------------------------------
    # Best ask
    # --------------------------------------------------------

    best_ask = {}

    for ask in asks:

        price = safe_float(
            ask.get("price"),
            0
        )

        volume = safe_float(
            ask.get("volume"),
            0
        )

        if price > 0 and volume > 0:

            best_ask = ask
            break

    # --------------------------------------------------------
    # Prices and quantities
    # --------------------------------------------------------

    bid_price = safe_float(
        best_bid.get("price"),
        0
    )

    bid_quantity = safe_float(
        best_bid.get("volume"),
        0
    )

    ask_price = safe_float(
        best_ask.get("price"),
        0
    )

    ask_quantity = safe_float(
        best_ask.get("volume"),
        0
    )

    # --------------------------------------------------------
    # Total order-book quantities
    # --------------------------------------------------------

    total_bid_quantity = safe_float(
        depth.get(
            "totalbuyqty",
            0
        ),
        0
    )

    total_ask_quantity = safe_float(
        depth.get(
            "totalsellqty",
            0
        ),
        0
    )

    # --------------------------------------------------------
    # Depth availability
    # --------------------------------------------------------

    depth_available = (
        bid_price > 0
        or
        ask_price > 0
        or
        total_bid_quantity > 0
        or
        total_ask_quantity > 0
    )

    # --------------------------------------------------------
    # Spread
    #
    # Only calculate spread when both sides exist.
    # --------------------------------------------------------

    spread_percent = None
    spread_pass = None

    if (
        bid_price > 0
        and
        ask_price > 0
        and
        ask_price >= bid_price
    ):

        midpoint = (
            bid_price + ask_price
        ) / 2

        if midpoint > 0:

            spread_percent = (
                (ask_price - bid_price)
                / midpoint
            ) * 100

            # No hard spread rejection here.
            #
            # We only calculate the value.
            # The screening logic can use it later.
            spread_pass = True

    # --------------------------------------------------------
    # Order-book liquidity
    #
    # This is informational only.
    # We do NOT reject the stock because one side
    # of the book is zero.
    # --------------------------------------------------------

    depth_liquidity_pass = (
        total_bid_quantity >=
        MIN_DEPTH_QUANTITY
        or
        total_ask_quantity >=
        MIN_DEPTH_QUANTITY
    )

    # --------------------------------------------------------
    # Extract result
    # --------------------------------------------------------

    result = {

        "depth_ltp":
            safe_float(
                depth.get("ltp"),
                0
            ),

        "ltq":
            safe_float(
                depth.get("ltq"),
                0
            ),

        "depth_volume":
            safe_float(
                depth.get("v"),
                0
            ),

        "depth_atp":
            safe_float(
                depth.get("atp"),
                0
            ),

        "bid_price":
            bid_price,

        "bid_quantity":
            bid_quantity,

        "ask_price":
            ask_price,

        "ask_quantity":
            ask_quantity,

        "total_bid_quantity":
            total_bid_quantity,

        "total_ask_quantity":
            total_ask_quantity,

        "spread_percent":
            spread_percent,

        "depth_available":
            depth_available,

        "spread_pass":
            spread_pass,

        "depth_liquidity_pass":
            depth_liquidity_pass
    }

    return result


# ============================================================
# APPLY LIQUIDITY FILTER
# ============================================================

def apply_liquidity_filter(
    fyers,
    df
):
    """
    Apply liquidity filtering.

    Primary liquidity condition:

        traded volume >= MIN_VOLUME

    Market depth is then collected for stocks
    that pass the volume condition.

    IMPORTANT:

    Market depth is informational here.

    A stock is NOT rejected merely because:

        total_bid_quantity == 0

    or:

        total_ask_quantity == 0
    """

    if df.empty:
        return df

    # --------------------------------------------------------
    # Volume filter
    # --------------------------------------------------------

    filtered = df[
        df["volume"].fillna(0)
        >= MIN_VOLUME
    ].copy()

    filtered.reset_index(
        drop=True,
        inplace=True
    )

    print(
        f"\nStocks passing volume "
        f"liquidity filter: "
        f"{len(filtered):,}"
    )

    if filtered.empty:
        return filtered

    # --------------------------------------------------------
    # Market depth
    # --------------------------------------------------------

    results = []

    total = len(filtered)

    print(
        f"\nChecking market depth "
        f"for {total:,} liquid stocks..."
    )

    for position, (
        index,
        row
    ) in enumerate(
        filtered.iterrows(),
        start=1
    ):

        symbol = row["symbol"]

        depth = get_market_depth(
            fyers,
            symbol
        )

        depth_data = extract_depth(
            depth
        )

        combined = row.to_dict()

        # ----------------------------------------------------
        # Depth available
        # ----------------------------------------------------

        if depth_data is not None:

            combined.update(
                depth_data
            )

        else:

            combined.update({

                "depth_ltp": None,

                "ltq": None,

                "depth_volume": None,

                "depth_atp": None,

                "bid_price": None,

                "bid_quantity": None,

                "ask_price": None,

                "ask_quantity": None,

                "total_bid_quantity": None,

                "total_ask_quantity": None,

                "spread_percent": None,

                "depth_available": False,

                "spread_pass": None,

                "depth_liquidity_pass": False
            })

        # ----------------------------------------------------
        # Volume liquidity passed
        # ----------------------------------------------------

        combined["liquidity_pass"] = True

        results.append(
            combined
        )

        print(
            f"\rProcessed "
            f"{position}/{total}",
            end=""
        )

        time.sleep(
            DEPTH_DELAY
        )

    print()

    return pd.DataFrame(
        results
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_market_data(df):
    """
    Save screened market data to CSV.
    """

    if df.empty:

        print(
            "\nNo market data to save."
        )

        return

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\nMarket data saved to:"
        f" {OUTPUT_FILE}"
    )


# ============================================================
# DISPLAY RESULTS
# ============================================================

def display_results(df):
    """
    Display the important columns from the
    final screened dataset.
    """

    if df.empty:

        print(
            "\nNo stocks currently satisfy "
            "the price and volume liquidity "
            "conditions."
        )

        return

    columns = [

        "symbol",

        "ltp",

        "volume",

        "change_percent",

        "bid_price",

        "bid_quantity",

        "ask_price",

        "ask_quantity",

        "total_bid_quantity",

        "total_ask_quantity",

        "spread_percent",

        "depth_available",

        "depth_liquidity_pass",

        "liquidity_pass"
    ]

    available_columns = [
        column
        for column in columns
        if column in df.columns
    ]

    print()
    print("=" * 100)
    print("FINAL SCREENED STOCKS")
    print("=" * 100)

    print(
        df[
            available_columns
        ].to_string(
            index=False
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AI/ML STOCK MARKET SCREENER")
    print("=" * 70)

    # --------------------------------------------------------
    # Load NSE symbols
    # --------------------------------------------------------

    print(
        "\n[1] Loading NSE equity universe..."
    )

    symbols = load_nse_equity_symbols()

    print(
        f"NSE equity universe: "
        f"{len(symbols):,} stocks"
    )

    if not symbols:

        print(
            "\nNo NSE symbols found."
        )

        return

    # --------------------------------------------------------
    # Connect to FYERS
    # --------------------------------------------------------

    print(
        "\n[2] Connecting to FYERS..."
    )

    fyers = create_fyers_client()

    print(
        "FYERS connection successful."
    )

    # --------------------------------------------------------
    # Get quotes
    # --------------------------------------------------------

    print(
        "\n[3] Downloading live quotes..."
    )

    quotes_df = get_quotes(
        fyers,
        symbols
    )

    print(
        f"\nReceived quotes for "
        f"{len(quotes_df):,} stocks."
    )

    if quotes_df.empty:

        print(
            "\nNo quotes received."
        )

        return

    # --------------------------------------------------------
    # Price filter
    # --------------------------------------------------------

    print(
        "\n[4] Applying price filter..."
    )

    ltp_df = filter_by_ltp(
        quotes_df
    )

    print(
        f"Stocks between "
        f"₹{MIN_LTP} and "
        f"₹{MAX_LTP}: "
        f"{len(ltp_df):,}"
    )

    if ltp_df.empty:

        print(
            "\nNo stocks passed "
            "the price filter."
        )

        return

    # --------------------------------------------------------
    # Liquidity filter
    # --------------------------------------------------------

    print(
        "\n[5] Applying liquidity filter..."
    )

    liquid_df = apply_liquidity_filter(
        fyers,
        ltp_df
    )

    print(
        "\nStocks passing "
        "price + volume liquidity:"
    )

    print(
        f"{len(liquid_df):,}"
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    print(
        "\n[6] Saving market data..."
    )

    save_market_data(
        liquid_df
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print(
        "\n[7] Displaying results..."
    )

    display_results(
        liquid_df
    )

    print()
    print("=" * 70)
    print("MARKET DATA SCREENING COMPLETE")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
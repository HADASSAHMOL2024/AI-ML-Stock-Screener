import time
import pandas as pd

from symbol_loader import load_nse_equity_symbols

from market_data import (
    create_fyers_client,
    get_quotes,
    filter_by_ltp,
    MIN_LTP,
    MAX_LTP
)

from market_depth import (
    get_market_depth,
    extract_depth_data
)


# ============================================================
# CONFIGURATION
# ============================================================

# Maximum number of stocks that may be used for
# market-depth processing if needed later.
MAX_DEPTH_STOCKS = 1262

# Small delay between depth requests.
DEPTH_DELAY = 0.05

# During development, only a limited number of
# stocks will be passed to historical analysis.
MAX_HISTORICAL_STOCKS = 20

# Number of historical days to use later.
HISTORICAL_DAYS = 60

# ML prediction threshold to be used later.
ML_THRESHOLD = 60.0

# Primary liquidity condition.
MIN_VOLUME = 1_000_000


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AI/ML STOCK MARKET SCREENER")
    print("=" * 70)

    # --------------------------------------------------------
    # STEP 1: Load NSE universe
    # --------------------------------------------------------

    print("\n[1] Loading NSE equity universe...")

    symbols = load_nse_equity_symbols()

    print(
        f"NSE equity universe: "
        f"{len(symbols):,} stocks"
    )

    if not symbols:
        print("\nNo NSE equity symbols found.")
        return

    # --------------------------------------------------------
    # STEP 2: Connect to FYERS
    # --------------------------------------------------------

    print("\n[2] Connecting to FYERS...")

    fyers = create_fyers_client()

    print("FYERS connection successful.")

    # --------------------------------------------------------
    # STEP 3: Get live quotes
    # --------------------------------------------------------

    print("\n[3] Downloading live quotes...")

    quotes_df = get_quotes(
        fyers,
        symbols
    )

    if quotes_df.empty:

        print(
            "\nNo quote data received."
        )

        return

    print(
        f"\nReceived quotes for "
        f"{len(quotes_df):,} stocks."
    )

    # --------------------------------------------------------
    # STEP 4: Price filter
    # --------------------------------------------------------

    print("\n[4] Applying price filter...")

    ltp_df = filter_by_ltp(
        quotes_df
    )

    print(
        f"Stocks between "
        f"₹{MIN_LTP} and ₹{MAX_LTP}: "
        f"{len(ltp_df):,}"
    )

    if ltp_df.empty:

        print(
            "\nNo stocks passed the price filter."
        )

        return

    # --------------------------------------------------------
    # STEP 5: Liquidity filter
    # --------------------------------------------------------

    print("\n[5] Applying liquidity filter...")

    # --------------------------------------------------------
    # Primary liquidity condition:
    #
    # traded volume >= 1,000,000
    #
    # Market depth is collected separately as additional
    # information.
    #
    # IMPORTANT:
    #
    # A stock is NOT rejected because:
    #
    #     total_bid_quantity == 0
    #
    # or:
    #
    #     total_ask_quantity == 0
    #
    # This is intentional because FYERS may return zero
    # on one side of the order book.
    # --------------------------------------------------------

    liquid_df = ltp_df[
        ltp_df["volume"].fillna(0) >= MIN_VOLUME
    ].copy()

    liquid_df.reset_index(
        drop=True,
        inplace=True
    )

    print(
        f"\nStocks passing volume "
        f"liquidity filter: "
        f"{len(liquid_df):,}"
    )

    if liquid_df.empty:

        print(
            "\nNo stocks currently satisfy "
            "the price and volume conditions."
        )

        return

    # --------------------------------------------------------
    # Collect market depth for volume-qualified stocks
    # --------------------------------------------------------

    print(
        f"\nChecking market depth for "
        f"{len(liquid_df):,} liquid stocks..."
    )

    depth_results = []

    total = len(liquid_df)

    for position, (index, row) in enumerate(
        liquid_df.iterrows(),
        start=1
    ):

        symbol = row["symbol"]

        # ----------------------------------------------------
        # Request market depth
        # ----------------------------------------------------

        depth = get_market_depth(
            fyers,
            symbol
        )

        # ----------------------------------------------------
        # Extract useful depth information
        # ----------------------------------------------------

        depth_data = extract_depth_data(
            depth
        )

        # Start with the quote information.
        combined = row.to_dict()

        # ----------------------------------------------------
        # Add depth information if available
        # ----------------------------------------------------

        if depth_data is not None:

            combined.update(
                depth_data
            )

        else:

            # Keep consistent columns when depth is unavailable.

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

                "spread_pass": None

            })

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # The stock has already passed the PRIMARY
        # liquidity condition based on traded volume.
        #
        # Therefore we DO NOT check:
        #
        #     depth_data["liquidity_pass"]
        #
        # here.
        #
        # Market depth is informational only.
        # ----------------------------------------------------

        combined["volume_liquidity_pass"] = True

        depth_results.append(
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

    # --------------------------------------------------------
    # Create final liquidity DataFrame
    # --------------------------------------------------------

    liquid_df = pd.DataFrame(
        depth_results
    )

    print(
        f"\nStocks passing liquidity filter: "
        f"{len(liquid_df):,}"
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if liquid_df.empty:

        print(
            "\nNo liquidity candidates were generated."
        )

        return

    # --------------------------------------------------------
    # STEP 6: Display liquidity candidates
    # --------------------------------------------------------

    print(
        "\nLiquidity candidates:"
    )

    display_columns = [

        "symbol",

        "ltp",

        "volume",

        "bid_price",
        "bid_quantity",

        "ask_price",
        "ask_quantity",

        "total_bid_quantity",
        "total_ask_quantity",

        "ltq",

        "depth_available",

        "spread_percent"

    ]

    available_columns = [
        column
        for column in display_columns
        if column in liquid_df.columns
    ]

    print(
        liquid_df[
            available_columns
        ].to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # STEP 7: Limit historical processing during development
    # --------------------------------------------------------

    historical_df = liquid_df.head(
        MAX_HISTORICAL_STOCKS
    ).copy()

    print(
        f"\nFor development, historical analysis "
        f"will be performed on "
        f"{len(historical_df)} stocks."
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Once the complete pipeline is verified, "
        "we will remove this development limit."
    )

    # --------------------------------------------------------
    # STEP 8: Save liquidity candidates
    # --------------------------------------------------------

    liquid_df.to_csv(
        "liquidity_candidates.csv",
        index=False
    )

    print(
        "\nLiquidity candidates saved to:"
    )

    print(
        "liquidity_candidates.csv"
    )

    # --------------------------------------------------------
    # STEP 9: Historical analysis placeholder
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "LIQUIDITY SCREENING COMPLETED"
    )

    print(
        "=" * 70
    )

    print(
        "\nNext stage:"
    )

    print(
        "Historical SMMA + ML analysis "
        "will be connected here."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()
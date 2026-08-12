# AI/ML Stock Market Screener

An automated Python-based stock market screening system designed to reduce the NSE equity universe to a smaller set of potentially suitable stocks using live market data, price filtering, traded-volume liquidity analysis, market-depth information, historical data processing, feature engineering, technical indicators, and machine learning.

The project integrates the **FYERS API** to obtain live NSE market information and follows a modular multi-stage screening pipeline.

---

## Project Overview

Analyzing thousands of stocks individually can be time-consuming. This project automates the initial screening process by progressively reducing the NSE equity universe using configurable market conditions.

The current screening pipeline consists of:

1. Loading the NSE equity universe
2. Authenticating with the FYERS API
3. Downloading live market quotes
4. Applying a price-based filter
5. Applying a traded-volume liquidity filter
6. Collecting market-depth information
7. Saving liquidity-qualified stocks
8. Selecting stocks for historical analysis
9. Performing historical data processing
10. Generating technical and engineered features
11. Applying machine learning analysis

The system is implemented as a modular Python application so that each component can be tested and improved independently.

---

## Objectives

The main objectives of the project are:

- Automate stock screening using live market information.
- Reduce the number of stocks requiring detailed analysis.
- Identify stocks within a configurable price range.
- Remove stocks with insufficient traded volume.
- Collect market-depth information for liquidity-qualified stocks.
- Prepare selected stocks for historical analysis.
- Generate technical and machine-learning features.
- Provide a modular framework for AI/ML-based stock screening.

---

## System Architecture

The project follows a sequential screening architecture:

```text
                NSE Equity Universe
                        |
                        v
              FYERS Symbol Master
                        |
                        v
                FYERS Authentication
                        |
                        v
                Live Market Quotes
                        |
                        v
                  Price Filter
                        |
                        v
              Traded Volume Filter
                        |
                        v
               Market Depth Data
                        |
                        v
            Liquidity Candidates
                        |
                        v
             Historical Market Data
                        |
                        v
              Feature Engineering
                        |
                        v
             Technical Indicators
                        |
                        v
              Machine Learning
                        |
                        v
                Screening Results



**Implementation completed so far**:
 The current version of the AI/ML Stock Market Screener successfully integrates with the FYERS API and loads the NSE equity universe from the FYERS symbol master. The system authenticates with FYERS using an authorization-code workflow and stores the generated access token locally for subsequent API requests. It retrieves live quote data for the NSE equity universe in batches and extracts important market attributes including LTP, trading volume, bid/ask prices, OHLC values, percentage change, and average traded price. A configurable price filter is then applied to reduce the stock universe, followed by a primary liquidity filter based on traded volume, currently configured at a minimum of 1,000,000 shares. For stocks passing the volume condition, market-depth information is additionally collected and stored as informational data without rejecting stocks solely because bid or ask depth is unavailable. The resulting liquidity-qualified stocks are saved to liquidity_candidates.csv for further processing. The project also contains separate modules for historical data handling, feature engineering, technical analysis, and machine-learning processing, providing the foundation for the subsequent historical SMMA and ML-based screening stages.


**Author

Hadassah Mol

M.Tech Computer Science (AI & Software Engineering)**

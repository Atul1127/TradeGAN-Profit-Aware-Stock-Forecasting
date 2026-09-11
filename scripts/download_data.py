"""Download adjusted stock and benchmark OHLC data into the local data directory."""

from pathlib import Path
import argparse

import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]


def download_data(
    metadata_file: Path,
    output_dir: Path,
    start_date: str = "2010-01-01",
    end_date: str = "2021-12-31",
    exchange_suffix: str = ".NS",
) -> None:
    metadata = pd.read_csv(metadata_file)
    output_dir.mkdir(parents=True, exist_ok=True)

    required_columns = {"ticker_x", "ticker_y"}
    missing = required_columns - set(metadata.columns)
    if missing:
        raise ValueError(f"Metadata file is missing columns: {sorted(missing)}")

    for _, row in metadata.iterrows():
        stock = str(row["ticker_x"])
        benchmark = str(row["ticker_y"])

        for ticker in (stock + exchange_suffix, benchmark):
            data = yf.download(
                ticker,
                start=start_date,
                end=end_date,
                progress=False,
                auto_adjust=True,
            )
            if data.empty:
                print(f"No data returned for {ticker}; skipping.")
                continue

            data = data[["Open", "Close"]].copy()
            data.columns = ["AdjOpen", "AdjClose"]
            data.reset_index(inplace=True)
            data.rename(columns={"Date": "date"}, inplace=True)
            data = data.dropna(subset=["date", "AdjOpen", "AdjClose"])
            data = data.sort_values("date").drop_duplicates("date", keep="last")

            base_ticker = ticker[:-len(exchange_suffix)] if ticker.endswith(exchange_suffix) else ticker
            output_file = output_dir / f"{base_ticker}.csv"
            data.to_csv(output_file, index=False)
            print(f"Saved {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download TradeGAN market data.")
    parser.add_argument("--metadata", type=Path, default=ROOT / "stocks-etfs-list.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data")
    parser.add_argument("--start", default="2010-01-01")
    parser.add_argument("--end", default="2021-12-31")
    args = parser.parse_args()
    download_data(args.metadata, args.output, args.start, args.end)

import polars as pl
import numpy as np

# =========================
# USER - CONFIG
# =========================

PARQUET_FILEPATH = "file_path to your parquet data file"
FAST_EMA = 20
SLOW_EMA = 50

# ==============================
# DATA LOADING USING LAZY FRAMES
# ==============================

lazy_frame = (
    pl.scan_parquet(PARQUET_FILEPATH)
    .select([
        "datetime",
        "ask_open",
        "ask_high",
        "ask_low",
        "ask_close",
        "bid_open",
        "bid_high",
        "bid_low",
        "bid_close",
    ])
) #We are only loading the required columns to conserve RAM and keep the program stable

# ==============================
# STRATEGY - LOGIC
# ==============================

lazy_frame = (
    lazy_frame
    .with_columns(
        mid_open = ((pl.col("ask_open") + pl.col("bid_open")) * 0.5),
        mid_high = ((pl.col("ask_high") + pl.col("bid_high")) * 0.5),
        mid_low = ((pl.col("ask_low") + pl.col("bid_low")) * 0.5),
        mid_close = ((pl.col("ask_close") + pl.col("bid_close")) * 0.5)
    )#Computing MID OHLC data to avoid SPREAD BIAS
    
    .with_columns([
        pl.col("mid_close").ewm_mean(span=FAST_EMA, adjust=False).alias("fast_ema"), #Coputing Fast EMA using VECTORIZATION
        pl.col("mid_close").ewm_mean(span=SLOW_EMA, adjust=False).alias("slow_ema") #Coputing Slow EMA using VECTORIZATION
    ])
    
    .with_columns([
        ((pl.col("fast_ema") > pl.col("slow_ema")) & (pl.col("fast_ema").shift(1) <= pl.col("slow_ema").shift(1))).alias("long_entry"), #Crossover Entry Detection
        ((pl.col("fast_ema") < pl.col("slow_ema")) & (pl.col("fast_ema").shift(1) >= pl.col("slow_ema").shift(1))).alias("long_exit")  #Crossover Exit Detection 
    ])

    .with_columns([
        pl.col("long_entry").shift(1).alias("exec_long_entry"),
        pl.col("long_exit").shift(1).alias("exec_long_exit")
    ])

    .with_columns([
        pl.when(pl.col("exec_long_entry"))
        .then(1)
        .when(pl.col("exec_long_exit"))
        .then(0)
        .otherwise(None)
        .alias("position_signal")
    ])
    
    .with_columns([
        pl.col("position_signal")
        .forward_fill()
        .fill_null(0)
        .alias("position_state")
    ])

    .with_columns(
        pl.col("mid_open").pct_change().alias("open_return")
    )
    .with_columns(
        (pl.col("open_return") * pl.col("position_state").shift(1)).alias("strategy_return")
    )
    
    .with_columns(
        pl.col("exec_long_entry")
        .cast(pl.Int64)
        .fill_null(0)
        .cum_sum()
        .alias("trade_id")
    )
) 

lf = lazy_frame.collect(engine = "streaming")

# ==============================
# TRADE EXTRACTION
# ==============================

entries = (
    lf
    .filter(pl.col("exec_long_entry") == True)
    .select([
        "trade_id",
        pl.col("datetime").alias("entry_time"),
        pl.col("mid_open").alias("entry_price")
    ])
)
exits = (
    lf
    .filter(pl.col("exec_long_exit") == True)
    .select([
        "trade_id",
        pl.col("datetime").alias("exit_time"),
        pl.col("mid_open").alias("exit_price")
    ])
)

trades = (
    entries
    .join(exits, on="trade_id", how="inner")
    .with_columns(
        (pl.col("exit_price") / pl.col("entry_price") - 1).alias("trade_return")
    )
)
trades.write_csv("ema_crossover_trades.csv")

# ==============================
# PERFORMANCE METRICS
# ==============================

# --- Total Trades ---
total_trades = trades.height


# --- Total Profit (Compounded from bar returns) ---
equity_df = (
    lf
    .select("strategy_return")
    .with_columns(
        (1 + pl.col("strategy_return"))
        .cum_prod()
        .alias("equity_curve")
    )
)

final_equity = equity_df.select(pl.col("equity_curve").last()).item()
total_profit = final_equity - 1


# --- Max Drawdown ---
equity_df = equity_df.with_columns(
    pl.col("equity_curve")
    .cum_max()
    .alias("running_max")
)

equity_df = equity_df.with_columns(
    ((pl.col("equity_curve") / pl.col("running_max")) - 1)
    .alias("drawdown")
)

max_drawdown = equity_df.select(pl.col("drawdown").min()).item()


# ==============================
# PRINT RESULTS
# ==============================

print("\n========== PERFORMANCE ==========")
print(f"Total Trades   : {total_trades}")
print(f"Total Profit   : {total_profit:.4%}")
print(f"Max Drawdown   : {max_drawdown:.4%}")
print("=================================\n")

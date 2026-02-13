Overview:

This repository contains experimental backtesting projects built using Polars with a focus on:
- Vectorized computation
- Time-aligned execution modeling
- Lookahead-bias prevention
- Long-horizon validation

The goal is to build clean, scalable research infrastructure for systematic strategy development rather than curve-fit profitable systems.

Core Principles:

- LazyFrame data loading for memory efficiency
- No loop-based position logic
- Explicit next-bar execution (I+1)
- Open-to-open compounding
- Reproducible, long-term testing
All strategies are evaluated with proper causal alignment and realistic return modeling.

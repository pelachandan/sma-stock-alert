# Stock Alert - Position Trading System

A comprehensive Python-based position trading system for identifying and managing long-term stock trades. Uses quantitative analysis of 3+ strategies to scan S&P 500 stocks and monitor positions with strict risk management.

**Performance**: $204.5K total return on backtests (2022-2026), RelativeStrength_Ranker_Position only, 48.5% win rate, 2.52R average

## Features

- 🤖 **Automated Scanning**: Scans S&P 500 daily for trading opportunities
- 📊 **Multiple Strategies**: 
  - RelativeStrength_Ranker (momentum-based)
  - High52_Position (breakout-based)
  - BigBase_Breakout (consolidation breakout)
- 📈 **Position Monitoring**: Real-time exit signal detection (stops, targets, time-based)
- 💰 **Risk Management**: Position sizing, max positions, per-strategy limits
- 📧 **Email Alerts**: Actionable signals sent directly to inbox
- 📉 **Backtesting**: Historical performance analysis with walk-forward testing
- 🛠️ **Position Manager**: Interactive CLI for manual position management
- 🔍 **Indicators**: Moving averages, Bollinger Bands, RSI, Stochastic

## Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/stock-alert.git
cd stock-alert

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure (see Configuration section)
cp .env.example .env
# Edit .env with your API keys and email settings
```

### Usage

#### Scan for New Opportunities
```bash
python scripts/scan.py

# Optional: skip email alerts
python scripts/scan.py --no-email

# Optional: only check market regime
python scripts/scan.py --regime-only

# Optional: only monitor existing positions
python scripts/scan.py --monitor-only
```

#### Monitor Open Positions
```bash
python scripts/monitor.py

# Monitor specific position
python scripts/monitor.py --ticker AAPL

# Summary only (no detailed signals)
python scripts/monitor.py --summary
```

#### Manage Positions
```bash
# Interactive menu
python scripts/manage_positions.py

# List all positions
python scripts/manage_positions.py --list
```

#### Run Backtest
```bash
# Default backtest (2 years)
python scripts/backtest.py

# Custom date range
python scripts/backtest.py --start-date 2022-01-01 --end-date 2023-12-31

# Single strategy
python scripts/backtest.py --strategy RelativeStrength_Ranker_Position

# Walk-forward testing
python scripts/backtest.py --walk-forward 252

# Save results
python scripts/backtest.py --output results.csv
```

#### Download Market Data
```bash
# Specific stocks
python scripts/download_data.py --tickers AAPL,MSFT,GOOGL

# All S&P 500
python scripts/download_data.py --sp500

# Indices
python scripts/download_data.py --indices QQQ,SPY,DIA

# Custom date range
python scripts/download_data.py --tickers AAPL --start-date 2020-01-01
```

## Project Structure

```
stock-alert/
├── src/                              # Main source code
│   ├── data/                         # Market data retrieval
│   │   ├── market.py                 # Data provider
│   │   └── validators.py             # Data validation
│   ├── strategies/                   # Trading strategies
│   │   ├── relative_strength.py       # RS Ranker strategy
│   │   ├── high_52_week.py            # High 52-week breakout
│   │   ├── big_base.py                # Big Base consolidation
│   │   └── base.py                    # Strategy interface
│   ├── scanning/                     # Signal generation
│   │   ├── scanner.py                 # Strategy execution
│   │   └── validator.py               # Signal validation
│   ├── position_management/          # Position tracking
│   │   ├── tracker.py                 # Position state
│   │   ├── monitor.py                 # Exit detection
│   │   └── executor.py                # Order execution
│   ├── analysis/                     # Backtesting & metrics
│   │   ├── backtest.py                # Historical testing
│   │   └── metrics.py                 # Performance metrics
│   ├── indicators/                   # Technical indicators
│   │   ├── technical.py               # MA, BB, etc.
│   │   └── momentum.py                # RSI, stochastic
│   ├── config/                       # Configuration
│   │   ├── settings.py                # Global settings
│   │   └── strategies.py              # Strategy params
│   ├── notifications/                # Alerts & emails
│   │   └── email.py                   # Email notifications
│   └── utils/                        # Utilities
│
├── scripts/                          # Entry point scripts
│   ├── scan.py                        # Live scanner
│   ├── backtest.py                    # Backtester
│   ├── monitor.py                     # Position monitor
│   ├── manage_positions.py            # Position manager
│   └── download_data.py               # Data downloader
│
├── tests/                            # Tests
│   ├── unit/                          # Unit tests
│   ├── integration/                   # Integration tests
│   ├── fixtures/                      # Test data & mocks
│   │   ├── sample_data.py             # Sample market data
│   │   └── mocks.py                   # Mock objects
│   └── conftest.py                    # Pytest config
│
├── docs/                             # Documentation
│   ├── ARCHITECTURE.md                # System design
│   ├── API.md                         # API reference
│   ├── CONTRIBUTING.md                # Contributing guide
│   └── README.md                      # This file
│
├── data/                             # Data files
│   ├── sp500_constituents.csv         # S&P 500 ticker list
│   └── open_positions.json            # Current positions
│
├── requirements.txt                  # Python dependencies
├── .env.example                      # Environment template
└── README.md                         # This file
```

## Configuration

### Environment Variables (.env)

```bash
# Email Configuration
EMAIL_FROM=your-email@gmail.com
EMAIL_RECIPIENTS=recipient1@example.com,recipient2@example.com
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
SMTP_PASSWORD=your-app-password

# Google Cloud Storage
GCS_BUCKET=your-bucket-name
```

### Strategy Configuration

Strategy parameters (entry filters, stop multipliers, targets, time limits) are stored privately in **GCS** (`config/settings.json`) and loaded at runtime. See `src/config/settings.py` for the full list of configurable parameters. New strategies that rely on external settings, such as `EMA_StackAlignment_Position`, require a matching config block (for example `ema_stack_alignment`) in that private settings file before you enable them.

## Documentation

- **[ARCHITECTURE.md](docs/ARCHITECTURE.md)** - System design and data flow
- **[API.md](docs/API.md)** - Complete API reference
- **[CONTRIBUTING.md](docs/CONTRIBUTING.md)** - Development guidelines

## Strategies

### RelativeStrength_Ranker_Position
Ranks stocks by relative strength against QQQ. Looks for momentum leaders with strong trend alignment across multiple timeframes.

**Entry:**
- Relative strength vs QQQ above threshold
- Price > MA50 > MA100 > MA200 (all rising)
- ADX filter for trend strength
- Market regime filter (QQQ bull confirmation)

**Exit:**
- ATR-based stop loss
- R-multiple profit target
- Maximum hold period (time stop)
- Trailing MA exit

### High52_Position
Identifies breakouts near 52-week highs with Bollinger Band confirmation.

**Entry:**
- Price near 52-week high
- Bollinger Bands tightening (consolidation)
- Breakout above upper band with volume surge

**Exit:**
- ATR-based stop loss
- R-multiple profit target
- Maximum hold period (time stop)

### BigBase_Breakout_Position
Finds large consolidation bases and trades breakouts from them.

**Entry:**
- Extended consolidation period
- Tight price range
- Breakout above consolidation with volume surge

**Exit:**
- ATR-based stop loss
- R-multiple profit target
- Maximum hold period (time stop)

## Testing

```bash
# Run all tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration tests only
pytest tests/integration/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html

# Specific test
pytest tests/unit/test_strategies.py::test_relative_strength -v

# Parallel execution
pytest tests/ -n 4
```

## Development

### Setup Development Environment

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Pre-commit hooks (optional)
pre-commit install

# Run code quality checks
flake8 src/ tests/
mypy src/
black src/ tests/
```

### Contributing

See [CONTRIBUTING.md](docs/CONTRIBUTING.md) for:
- Code style guide
- Testing requirements
- Pull request process
- Adding new strategies/indicators

## Performance Metrics

From 4-year backtest (2022-2026):

| Metric | Value |
|--------|-------|
| Total Trades | 119 |
| Win Rate | 48.5% |
| Profit Factor | 2.52 |
| Total Return | 493.7% |
| Starting Capital | $100,000 |
| Ending Capital | $593,650 |
| Annualized Return | ~48.8% |
| Max Consecutive Wins | 5 |
| Max Drawdown | -18% |
| Avg Winner | $5,200 |
| Avg Loser | -$2,050 |

## Limitations

- **Paper Trading Only**: No real order execution (requires broker integration)
- **Data Delays**: Uses daily data (intraday signals not supported)
- **Past Performance**: Backtest results don't guarantee future performance
- **Market Dependent**: Strategies optimized for bull markets

## Future Enhancements

- [ ] Real-time data streaming
- [ ] Order execution (Interactive Brokers, Alpaca)
- [ ] Dynamic position sizing based on volatility
- [ ] Machine learning signal filtering
- [ ] Web dashboard for monitoring
- [ ] Mobile app notifications
- [ ] Multi-timeframe analysis
- [ ] Portfolio correlation tracking

## Troubleshooting

### No signals found
- Check market regime (might be bearish)
- Verify data is downloading correctly
- Check strategy parameters

### Email not sending
- Verify SMTP credentials in .env
- Check firewall/proxy settings
- Gmail users: use App Password, not regular password

### Tests failing
- Ensure all dependencies installed: `pip install -r requirements-dev.txt`
- Check Python version (3.8+)
- Run `pytest --tb=short` for detailed errors

### Slow backtests
- Use `--quiet` flag to skip detailed output
- Run walk-forward with smaller windows
- Reduce ticker list for testing

## Support

- **Issues**: GitHub Issues for bugs and feature requests
- **Discussions**: GitHub Discussions for questions
- **Email**: development@stock-alert.local

## License

MIT License - see LICENSE file for details

## Disclaimer

This system is for educational and research purposes. Past performance does not guarantee future results. Use at your own risk. Always do your own research before trading.

---

**Made with ❤️ by traders, for traders**

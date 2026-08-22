# BTC Live Contract

The first production scope is intentionally narrow: Binance Spot `BTCUSDT`, daily,
long-only, without margin or leverage. A completed UTC candle may produce a target
allocation, but the target is not an order until independent data and risk checks pass.

`config/live_btc.yaml` is the machine-readable contract. Production changes to its
venue, timing, capital limits, execution algorithm, or safety thresholds require a new
version and a new forward-validation campaign.

Initial live execution is supervised. Automatic promotion is prohibited until the
prospective evidence and operational gates are satisfied. The system must fail closed
on stale data, unresolved exchange state, balance mismatch, excessive slippage, or a
manual kill switch.

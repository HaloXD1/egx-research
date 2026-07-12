# TradingView workflow

The `egx tv` commands keep research local and reproducible. TradingView is an optional chart, Pine, and alert surface; it is not the project’s bulk-data or execution engine.

## Complete CLI surface

| Area | Commands | Output or purpose |
| --- | --- | --- |
| Readiness | `doctor`, `status`, `audit` | Validate config, registry, Pine metadata, symbol mappings, data freshness, and execution assumptions. |
| Pine lifecycle | `strategy list`, `show`, `templates`, `new`, `validate`, `export`, `params-export`, `inputs-compare`, `compile-record`, `promote` | Inventory scripts, synchronize inputs, record server compilation, export handoffs, and promote passing candidates. |
| Data | `sync`, `status`, `scan --refresh` | Refresh supported local feeds and prevent accidental work on stale data. |
| Research | `scan`, `backtest`, `batch-backtest`, `validate`, `batch-validate`, `parity`, `trade-parity` | Produce signals, benchmarks, walk-forward/holdout/cost stress, and event/trade-level Python-vs-Pine comparisons. |
| Operations | `paper-track`, `report`, `notify`, `schedule run`, `schedule cron` | Maintain paper signals, reports, daily pipelines, cron definitions, and notifications. |
| Run lifecycle | `run list`, `show`, `compare`, `archive`, `clean` | Inspect, compare, retain, and explicitly clean generated runs. |
| Webhooks | `webhook verify`, `webhook serve` | Verify HMAC signatures, deduplicate event IDs, and persist accepted payloads. |
| TradingView account | `strategy export`, `chart open`, `alert prepare`, `alert create`, `account prepare`, `account apply` | Package Pine and execute reviewed browser steps using a user-owned authenticated profile. |

Run the readiness gate before a handoff:

```bash
egx tv doctor
egx tv doctor --strict-data --run-id tv-readiness
egx tv strategy export --strategy crypto_donchian_breakout --symbol BTCUSDT --run-id tv-btc-export
```

`doctor` reports Pine/registry mismatches as errors. Missing or stale data is a warning by default and an error with `--strict-data`. Export bundles include the exact Pine source, source hash, strategy parameters, symbol mapping, execution model, signal contract, and audit result.

## Local workflow

```bash
egx tv strategy list
egx tv strategy show crypto_donchian_breakout
egx tv strategy templates
egx tv strategy validate --path tradingview/btc_donchian_breakout_strategy.pine
egx tv strategy validate --strategy crypto_donchian_breakout
egx tv scan --strategy crypto_donchian_breakout --run-id tv-scan-btc
egx tv backtest --strategy crypto_donchian_breakout --symbol BTCUSDT --run-id tv-btc-backtest
egx tv report --run-id tv-btc-backtest
egx tv status --symbol BTCUSDT
egx tv scan --strategy crypto_donchian_breakout --symbol BTCUSDT --refresh --run-id tv-btc-latest
egx tv validate --strategy crypto_donchian_breakout --symbol BTCUSDT --run-id tv-btc-validation
egx tv paper-track --strategy crypto_donchian_breakout --symbol BTCUSDT --run-id tv-btc-paper
```

Generated files are written under `runs/`. The registry and symbol map live under `tradingview/`; fill in uncertain TradingView symbols locally rather than assuming an exchange mapping.

## Signal parity

Export signal or plot data from TradingView to CSV, then compare it with the local Python adapter:

```bash
egx tv parity \
  --strategy crypto_donchian_breakout \
  --symbol BTCUSDT \
  --pine-events /path/to/tradingview-events.csv \
  --run-id tv-parity-btc
```

Parity compares confirmed signal dates and actions. It does not claim that TradingView’s broker emulator, fills, spread, or account equity are identical to the local backtester.

Trading-bar tolerance uses the local dataset calendar, so a one-bar Friday-to-Monday shift counts as one bar rather than three calendar days. Compare Strategy Tester trades separately:

```bash
egx tv trade-parity \
  --local-trades runs/tv-btc-backtest/trades.csv \
  --tradingview-trades /path/to/strategy-tester.csv \
  --output runs/tv-btc-backtest/trade_parity.json
```

Backtest artifacts include monthly DCA and buy-and-hold benchmarks. Crypto runs also include weekly DCA.

## Parameters, compilation, and promotion

```bash
egx tv strategy params-export --strategy crypto_donchian_breakout --output-dir runs/tv-inputs
egx tv strategy inputs-compare --strategy crypto_donchian_breakout --tradingview-inputs /path/to/inputs.csv
egx tv strategy compile-record --strategy crypto_donchian_breakout --status pass --run-id tv-compile --evidence /path/to/screenshot.png
egx tv strategy promote --strategy crypto_donchian_breakout --validation-run-id tv-btc-validation --status validated --confirm
```

Promotion edits the registry only when the selected validation run belongs to the same strategy and has status `pass`. JSON schemas for registry, symbols, and browser requests live under `tradingview/schemas/`.

## Operational checks

```bash
egx tv sync --symbol BTCUSDT
egx tv audit --strategy crypto_donchian_breakout
egx tv notify --run-id tv-btc-paper
egx tv notify --run-id tv-btc-paper --send --channel slack
egx tv alert prepare --strategy crypto_donchian_breakout --symbol BTCUSDT --run-id tv-alert-review
egx tv schedule run --strategy crypto_donchian_breakout --symbol BTCUSDT --run-id tv-daily --refresh
egx tv schedule cron --strategy crypto_donchian_breakout --symbol BTCUSDT --hour 18
```

Validation produces walk-forward windows, holdout results, and fee/slippage stress results. Paper tracking produces `paper_log.csv` and `current_signal.json`. Notifications are dry-run by default. Supported channels are generic webhook, Slack, Discord, Telegram, and SMTP email.

Webhook verification requires a SHA-256 HMAC signature:

```bash
export EGX_TV_WEBHOOK_SECRET='replace-me'
egx tv webhook serve --host 127.0.0.1 --port 8765
```

Send the signature in `X-EGX-Signature` as either the hex digest or `sha256=<digest>`. Accepted events are idempotent by `id`, with a payload hash fallback.

## Safety boundary

Full chart application and alert creation are intentionally guarded. The local workflow does not use TradingView as the source for reliable bulk data or live order execution. Keep credentials, browser profiles, cookies, and session state outside the repository.

The optional chart snapshot command uses Playwright:

```bash
pip install -e '.[tradingview-browser]'
egx tv chart open --url 'https://www.tradingview.com/chart/' --screenshot runs/chart.png
```

Authenticated account operations require all of the following:

1. Install `.[tradingview-browser]` and its Chromium runtime.
2. Keep the browser profile outside the repository.
3. Generate and review an account request containing explicit `ui_steps`.
4. Set `EGX_TV_ALLOW_ACCOUNT_MUTATIONS=1`.
5. Pass `--confirm` and an artifact directory.

```bash
egx tv account prepare --operation upload_script --ui-steps-json '[{"action":"goto","url":"https://www.tradingview.com/chart/"}]' --output /tmp/tv-request.json
EGX_TV_ALLOW_ACCOUNT_MUTATIONS=1 egx tv account apply --request-file /tmp/tv-request.json --profile-dir /path/outside/repo/tv-profile --artifact-dir runs/tv-account --confirm
```

Allowed steps are `goto`, `click`, `fill`, `select`, `upload`, and `wait`. Every apply records completed steps, a final/failure screenshot, and a Playwright trace. Requests with `live_order_execution: true` are rejected.

## Credentials

Use environment variables only:

| Integration | Variables |
| --- | --- |
| Generic/Slack/Discord | `EGX_TV_WEBHOOK_URL` |
| Signed receiver | `EGX_TV_WEBHOOK_SECRET` |
| Telegram | `EGX_TV_TELEGRAM_TOKEN`, `EGX_TV_TELEGRAM_CHAT_ID` |
| Email | `EGX_TV_SMTP_HOST`, `EGX_TV_SMTP_PORT`, `EGX_TV_SMTP_USERNAME`, `EGX_TV_SMTP_PASSWORD`, `EGX_TV_EMAIL_FROM`, `EGX_TV_EMAIL_TO` |
| Browser mutations | `EGX_TV_ALLOW_ACCOUNT_MUTATIONS=1` plus an external browser profile |

## Exit behavior

- `0`: command completed or a dry run was generated.
- `1`: validation, doctor, event parity, trade parity, or input parity failed.
- `2`: invalid CLI arguments or a missing safety confirmation.

## Deliberate boundaries

The CLI records TradingView compilation evidence but cannot compile Pine on TradingView servers without an authenticated TradingView session. UI steps are intentionally externalized because selectors can change. The repository does not scrape TradingView as a bulk-data provider, store credentials/cookies, connect to brokers, or place live orders.

# Crypto Live Operations Runbook

## Normal daily sequence

1. Confirm the UTC daily candle is complete and independently validated.
2. Reproduce the frozen production signal and compare it with research output.
3. Reconcile balances, open orders, and every non-terminal local order.
4. Run pre-trade risk checks. Any failed check blocks execution.
5. In supervised mode, review and approve the order intent.
6. Submit once using the deterministic client order ID.
7. Reconcile fills, fees, balances, and target allocation.
8. Store the decision, order, fill, accounting, and health records.

## Unknown order state

Never resubmit an order after a timeout or server error. Query by the existing client
order ID until the exchange state is known. Engage the kill switch if state cannot be
resolved before another decision is due.

## Emergency stop

Create the configured kill-switch file, stop new submissions, reconcile all existing
orders, cancel orders only after their state is known, and verify balances through the
read-only key. Manual liquidation is a separate, explicitly approved action.

## Recovery

Restore the latest verified state backup, open the persistent order database, reconcile
against the exchange before processing new signals, and record the recovery incident.
Production remains blocked while any orphan order or balance mismatch exists.

## Credential policy

The trading key must not permit withdrawals and must use an IP allowlist. Monitoring
uses a separate read-only key. Secrets belong in the runtime secret store, never in
configuration files, command history, manifests, reports, or logs.

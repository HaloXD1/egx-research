from __future__ import annotations

import hashlib
import hmac
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Any, Protocol
from urllib.parse import urlencode

import requests


class ExchangeError(RuntimeError):
    pass


class UnknownExecutionState(ExchangeError):
    pass


@dataclass(frozen=True)
class OrderRequest:
    intent_id: str
    symbol: str
    side: str
    quantity: float
    order_type: str = "MARKET"
    price: float | None = None


@dataclass(frozen=True)
class ExchangeOrder:
    client_order_id: str
    exchange_order_id: str
    symbol: str
    side: str
    status: str
    original_quantity: float
    executed_quantity: float
    cumulative_quote_quantity: float


@dataclass(frozen=True)
class SymbolFilters:
    quantity_step: float
    minimum_quantity: float
    price_tick: float
    minimum_notional: float


class ExchangeAdapter(Protocol):
    def submit_order(
        self, request: OrderRequest, client_order_id: str
    ) -> ExchangeOrder: ...

    def get_order(self, symbol: str, client_order_id: str) -> ExchangeOrder: ...

    def cancel_order(self, symbol: str, client_order_id: str) -> ExchangeOrder: ...

    def open_orders(self, symbol: str | None = None) -> list[ExchangeOrder]: ...

    def balances(self) -> dict[str, float]: ...


def deterministic_client_order_id(intent_id: str) -> str:
    digest = hashlib.sha256(intent_id.encode("utf-8")).hexdigest()[:24]
    return f"egx-{digest}"


def sign_query(params: dict[str, Any], secret: str) -> str:
    query = urlencode(params)
    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()


def _round_step(value: float, step: float) -> float:
    if step <= 0:
        raise ValueError("exchange step must be positive")
    units = Decimal(str(value)) / Decimal(str(step))
    return float(units.quantize(Decimal("1"), rounding=ROUND_DOWN) * Decimal(str(step)))


def conform_order(
    request: OrderRequest,
    filters: SymbolFilters,
    reference_price: float,
) -> OrderRequest:
    quantity = _round_step(request.quantity, filters.quantity_step)
    if quantity < filters.minimum_quantity:
        raise ValueError("order quantity is below the exchange minimum")
    if quantity * reference_price < filters.minimum_notional:
        raise ValueError("order notional is below the exchange minimum")
    price = (
        _round_step(request.price, filters.price_tick)
        if request.price is not None
        else None
    )
    return OrderRequest(
        intent_id=request.intent_id,
        symbol=request.symbol,
        side=request.side,
        quantity=quantity,
        order_type=request.order_type,
        price=price,
    )


class BinanceSpotAdapter:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        *,
        base_url: str = "https://testnet.binance.vision",
        timeout_seconds: float = 10.0,
        session: requests.Session | None = None,
    ) -> None:
        if not api_key or not api_secret:
            raise ValueError("Binance API key and secret are required")
        self._api_key = api_key
        self._api_secret = api_secret
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    def _public_request(self, path: str, params: dict[str, Any]) -> Any:
        try:
            response = self.session.get(
                f"{self.base_url}{path}",
                params=params,
                timeout=self.timeout_seconds,
            )
        except requests.RequestException as exc:
            raise ExchangeError("exchange public request failed") from exc
        if not response.ok:
            raise ExchangeError(
                f"exchange public request returned {response.status_code}"
            )
        return response.json()

    def symbol_filters(self, symbol: str) -> SymbolFilters:
        payload = self._public_request("/api/v3/exchangeInfo", {"symbol": symbol})
        symbols = payload.get("symbols", [])
        if len(symbols) != 1:
            raise ExchangeError(f"exchange did not return unique filters for {symbol}")
        filters = {
            item["filterType"]: item for item in symbols[0].get("filters", [])
        }
        market_lot = filters.get("MARKET_LOT_SIZE")
        lot = (
            market_lot
            if market_lot and float(market_lot.get("stepSize", 0.0)) > 0
            else filters.get("LOT_SIZE")
        )
        price = filters.get("PRICE_FILTER")
        notional = filters.get("NOTIONAL") or filters.get("MIN_NOTIONAL")
        if not lot or not price or not notional:
            raise ExchangeError(f"exchange filters are incomplete for {symbol}")
        minimum_notional = notional.get("minNotional", notional.get("notional"))
        return SymbolFilters(
            quantity_step=float(lot["stepSize"]),
            minimum_quantity=float(lot["minQty"]),
            price_tick=float(price["tickSize"]),
            minimum_notional=float(minimum_notional),
        )

    def server_time(self) -> int:
        return int(self._public_request("/api/v3/time", {})["serverTime"])

    def _signed_request(
        self,
        method: str,
        path: str,
        params: dict[str, Any],
    ) -> Any:
        payload = {
            **params,
            "timestamp": int(time.time() * 1000),
            "recvWindow": 5000,
        }
        payload["signature"] = sign_query(payload, self._api_secret)
        try:
            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                params=payload,
                headers={"X-MBX-APIKEY": self._api_key},
                timeout=self.timeout_seconds,
            )
        except requests.Timeout as exc:
            raise UnknownExecutionState("exchange request timed out") from exc
        except requests.RequestException as exc:
            raise ExchangeError("exchange request failed") from exc
        if response.status_code >= 500:
            raise UnknownExecutionState(
                f"exchange returned {response.status_code}; execution state is unknown"
            )
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After", "unknown")
            raise ExchangeError(f"exchange rate limit; retry after {retry_after}")
        if not response.ok:
            raise ExchangeError(
                f"exchange rejected request with status {response.status_code}"
            )
        return response.json()

    @staticmethod
    def _parse_order(payload: dict[str, Any]) -> ExchangeOrder:
        return ExchangeOrder(
            client_order_id=str(payload.get("clientOrderId", "")),
            exchange_order_id=str(payload.get("orderId", "")),
            symbol=str(payload["symbol"]),
            side=str(payload["side"]).lower(),
            status=str(payload["status"]).lower(),
            original_quantity=float(payload.get("origQty", 0.0)),
            executed_quantity=float(payload.get("executedQty", 0.0)),
            cumulative_quote_quantity=float(payload.get("cummulativeQuoteQty", 0.0)),
        )

    def submit_order(
        self, request: OrderRequest, client_order_id: str
    ) -> ExchangeOrder:
        params: dict[str, Any] = {
            "symbol": request.symbol,
            "side": request.side.upper(),
            "type": request.order_type.upper(),
            "quantity": format(request.quantity, ".12f").rstrip("0").rstrip("."),
            "newClientOrderId": client_order_id,
            "newOrderRespType": "FULL",
        }
        if request.price is not None:
            params.update({"price": request.price, "timeInForce": "GTC"})
        return self._parse_order(self._signed_request("POST", "/api/v3/order", params))

    def get_order(self, symbol: str, client_order_id: str) -> ExchangeOrder:
        payload = self._signed_request(
            "GET",
            "/api/v3/order",
            {"symbol": symbol, "origClientOrderId": client_order_id},
        )
        return self._parse_order(payload)

    def cancel_order(self, symbol: str, client_order_id: str) -> ExchangeOrder:
        payload = self._signed_request(
            "DELETE",
            "/api/v3/order",
            {"symbol": symbol, "origClientOrderId": client_order_id},
        )
        return self._parse_order(payload)

    def open_orders(self, symbol: str | None = None) -> list[ExchangeOrder]:
        params = {"symbol": symbol} if symbol else {}
        payload = self._signed_request("GET", "/api/v3/openOrders", params)
        return [self._parse_order(item) for item in payload]

    def balances(self) -> dict[str, float]:
        payload = self._signed_request("GET", "/api/v3/account", {})
        return {
            str(item["asset"]): float(item["free"]) + float(item["locked"])
            for item in payload.get("balances", [])
        }

    def describe(self) -> dict[str, Any]:
        return {
            "adapter": type(self).__name__,
            "base_url": self.base_url,
            "timeout_seconds": self.timeout_seconds,
        }

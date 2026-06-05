"""SQL tools exposed as external functions to the Monty CodeMode sandbox.

The agent writes Python in `run_code` (via CodeMode) that calls these SQL
functions on a pre-seeded SQLite database. Each eval gets its own isolated
in-memory Database, so cases never share mutable state.

The seed embeds patterns that multi-step data-science tasks can exploit:
- Customers signed up across ~18 months → enables cohort/retention analysis.
- Wide CLV variance → enables segmentation tasks.
- Some products with low stock + high velocity → exposes reorder-priority work.
- A handful of intentional pricing anomalies → gives anomaly-investigation
  tasks something real to find.
- A subgroup of customers inactive for >90 days → enables churn scoring.
"""

from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import logfire


class SchemaMismatchError(Exception):
    """Raised and immediately caught inside ``query`` to record an *exception*
    span when SQL hits the migrated-away schema.

    It never propagates into the sandbox — letting it escape a ``logfire.span``
    is just how we get Logfire to mark that span ``is_exception=true`` (so the
    failure shows as a red error in the trace and is countable on dashboards),
    while the agent still recovers from the returned error dict exactly as
    before.
    """


# ---------------------------------------------------------------------------
# Database container
# ---------------------------------------------------------------------------


@dataclass
class Database:
    """In-memory SQLite database for a single eval case.

    ``check_same_thread=False`` is required because pydantic-ai dispatches
    sync tool functions via run_in_executor (thread-pool workers), while
    the connection itself is created on the main async thread. Each eval
    gets its own isolated Database instance, so there is no concurrent
    write risk.
    """

    _conn: sqlite3.Connection = field(
        default_factory=lambda: sqlite3.connect(":memory:", check_same_thread=False)
    )

    def __post_init__(self) -> None:
        self._conn.row_factory = sqlite3.Row

    @property
    def conn(self) -> sqlite3.Connection:
        return self._conn


# ---------------------------------------------------------------------------
# SQL tool factory
# ---------------------------------------------------------------------------


def make_sql_tools(db: Database) -> dict[str, Any]:
    """Build the SQL tools dict for a given Database.

    The returned callables become external functions in the Monty sandbox;
    the model writes Python that calls them.
    """

    def query(sql: str) -> list[dict[str, Any]]:
        """Execute a SQL statement and return rows as a list of dicts.

        Supports SELECT, INSERT, UPDATE, DELETE, CREATE TABLE, etc.
        SELECTs return rows as ``[{column: value}, ...]``.
        Writes return ``[{"rows_affected": N}]``.
        """
        cursor = db.conn.cursor()
        try:
            cursor.execute(sql)
            if cursor.description:
                columns = [col[0] for col in cursor.description]
                return [dict(zip(columns, row, strict=False)) for row in cursor.fetchall()]
            db.conn.commit()
            return [{"rows_affected": cursor.rowcount}]
        except Exception as e:
            message = f"{type(e).__name__}: {e}"
            # "no such table" / "no such column" is the fingerprint of a schema
            # the system prompt describes wrongly (see the migration note on
            # ``seed_database``). We surface those as error-level EXCEPTION spans
            # so they're visible (red) in the trace, countable on dashboards
            # (is_exception), and strong *failure* evidence for the optimizer
            # (its candidate query treats level>=error / is_exception as a
            # failure; ``failure_reason`` becomes the one-line summary). The
            # agent still recovers from the returned error dict, so the
            # conversation transcript shows both the break and the fix.
            #
            # Incidental SQL errors (an agent typo, etc.) stay at WARNING so the
            # post-fix "after" state shows ~zero errors.
            lowered = str(e).lower()
            schema_mismatch = "no such table" in lowered or "no such column" in lowered
            if schema_mismatch:
                # Record a real EXCEPTION span: let SchemaMismatchError escape a
                # logfire.span (so Logfire marks it is_exception=true / red in
                # the trace and it's countable as an error), then swallow it
                # here. The agent never sees the exception — it gets the error
                # dict below and recovers exactly as before.
                try:
                    with logfire.span(
                        "SQL failed against the live schema: {message}",
                        message=message,
                        sql=sql,
                        _level="error",
                        failure_reason=f"{message} — the prompt's documented schema is stale",
                        likely_cause="schema_mismatch",
                    ):
                        raise SchemaMismatchError(message)
                except SchemaMismatchError:
                    pass
            else:
                logfire.warning("SQL query failed: {message}", message=message, sql=sql)
            return [{"error": message}]

    def list_tables() -> list[str]:
        """List all tables in the database."""
        cursor = db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
        return [row[0] for row in cursor.fetchall()]

    def describe_table(name: str) -> list[dict[str, str | bool]]:
        """Describe a table's schema (column name, type, nullable, primary key)."""
        cursor = db.conn.cursor()
        try:
            cursor.execute(f"PRAGMA table_info({name})")
            return [
                {
                    "name": row[1],
                    "type": row[2],
                    "nullable": not row[3],
                    "primary_key": bool(row[5]),
                }
                for row in cursor.fetchall()
            ]
        except Exception as e:
            return [{"error": f"{type(e).__name__}: {e}"}]

    def insert_rows(table: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Insert multiple rows into ``table``. ``rows`` is a list of column-value dicts."""
        if not rows:
            return {"inserted": 0}
        columns = list(rows[0].keys())
        placeholders = ", ".join(["?"] * len(columns))
        col_str = ", ".join(columns)
        sql = f"INSERT INTO {table} ({col_str}) VALUES ({placeholders})"  # noqa: S608
        try:
            db.conn.executemany(sql, [tuple(r.get(c) for c in columns) for r in rows])
            db.conn.commit()
            return {"inserted": len(rows)}
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}

    def table_count(table: str) -> int:
        """Return the row count of ``table`` (or -1 on error)."""
        cursor = db.conn.cursor()
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")  # noqa: S608
            return cursor.fetchone()[0]
        except Exception:
            return -1

    return {
        "query": query,
        "list_tables": list_tables,
        "describe_table": describe_table,
        "insert_rows": insert_rows,
        "table_count": table_count,
    }


# ---------------------------------------------------------------------------
# Seed data — richer e-commerce schema engineered for multi-step DS tasks
# ---------------------------------------------------------------------------

# A fixed seed makes every eval run see the same data, so prompts can be
# scored against deterministic ground truth.
_SEED = 42

# Schema constants
_STATES = ["CA", "NY", "TX", "WA", "FL", "IL", "MA", "GA"]
_SEGMENTS = ["consumer", "small_biz", "enterprise"]
_PRODUCT_CATEGORIES = ["Electronics", "Books", "Clothing", "Office", "Home"]

# Reference date — the "today" against which days-since-last-order is measured.
REFERENCE_DATE = date(2024, 7, 1)


# Customer-attributes migration: ``state`` and ``segment`` used to be plain
# columns; they now live — together with new fields Tom added (``region`` and a
# nested ``acquisition`` object) — inside a JSON-serialized ``attributes`` TEXT
# column. ``describe_table`` only reveals ``attributes TEXT``, NOT its keys, so
# an agent must SELECT rows and json.loads / json_extract them to discover where
# the data went, several follow-up queries deep. See the migration note on
# ``seed_database`` and the demo prompts.
_STATE_TO_REGION = {
    "CA": "West",
    "WA": "West",
    "TX": "South",
    "FL": "South",
    "GA": "South",
    "NY": "Northeast",
    "MA": "Northeast",
    "IL": "Midwest",
}
_ACQUISITION_CHANNELS = ["referral", "organic", "paid_search", "partner"]
_ACQUISITION_CAMPAIGNS = ["q1-launch", "spring-promo", "retargeting", "none"]


def _customer_attributes_json(cid: int, state: str, segment: str) -> str:
    """Build the JSON ``attributes`` blob for a customer (deterministic by id)."""
    return json.dumps(
        {
            "state": state,
            "segment": segment,
            "region": _STATE_TO_REGION[state],
            "acquisition": {
                "channel": _ACQUISITION_CHANNELS[cid % len(_ACQUISITION_CHANNELS)],
                "campaign": _ACQUISITION_CAMPAIGNS[(cid * 7) % len(_ACQUISITION_CAMPAIGNS)],
            },
        }
    )


def _build_customers(rng: random.Random) -> list[tuple]:
    """50 customers signed up across 18 months, segmented by state/segment."""
    rows: list[tuple] = []
    earliest = date(2023, 1, 1)
    for i in range(1, 51):
        offset_days = rng.randint(0, 540)
        signup = earliest + timedelta(days=offset_days)
        state = rng.choice(_STATES)
        segment = rng.choices(_SEGMENTS, weights=[0.6, 0.3, 0.1])[0]
        rows.append(
            (
                i,
                f"Customer {i:02d}",
                f"customer{i:02d}@example.com",
                state,
                segment,
                str(signup),
            )
        )
    return rows


def _build_products(rng: random.Random) -> list[tuple]:
    """25 products across 5 categories with mixed stock urgency.

    Some products are deliberately low-stock + popular, so reorder-priority
    tasks have real signal. Others are well-stocked + slow movers.
    """
    rows: list[tuple] = []
    name_idx = 0
    base_prices = {
        "Electronics": (29.0, 199.0),
        "Books": (12.0, 49.0),
        "Clothing": (15.0, 89.0),
        "Office": (5.0, 39.0),
        "Home": (19.0, 149.0),
    }
    for cat in _PRODUCT_CATEGORIES:
        per_cat = 5
        for _ in range(per_cat):
            name_idx += 1
            low, high = base_prices[cat]
            price = round(rng.uniform(low, high), 2)
            # Mix of stock urgency: some intentionally low + below threshold
            if rng.random() < 0.25:
                stock = rng.randint(2, 15)
                threshold = rng.randint(15, 30)
            else:
                stock = rng.randint(40, 300)
                threshold = rng.randint(20, 60)
            rows.append((name_idx, f"{cat} Item {name_idx:02d}", cat, price, stock, threshold))
    return rows


def _build_orders(
    rng: random.Random,
    customers: list[tuple],
    products: list[tuple],
) -> list[tuple]:
    """~300 orders, with deliberate patterns:

    - Most customers order 3-10 times. ~20% are one-shot.
    - ~20% of customers go silent after ~90+ days (churn signal).
    - 3 hand-built pricing anomalies (effective unit price differs from catalog).
    - One orphaned order pointing to a non-existent customer.
    - Order volume skews toward Electronics in CA, Books in NY (subgroup pattern).
    """
    rows: list[tuple] = []
    order_id = 0
    prod_by_id = {p[0]: p for p in products}

    for cust in customers:
        cid, _name, _email, state, _segment, signup_str = cust
        signup = date.fromisoformat(signup_str)
        # Order frequency: most customers active, some churned
        is_churned = rng.random() < 0.20
        is_one_shot = rng.random() < 0.20

        if is_one_shot:
            n_orders = 1
        elif is_churned:
            n_orders = rng.randint(2, 5)
        else:
            n_orders = rng.randint(3, 12)

        # Customer's preferred category — state-conditioned for subgroup signal
        if state == "CA":
            pref_cat = rng.choices(_PRODUCT_CATEGORIES, weights=[0.45, 0.10, 0.15, 0.10, 0.20])[0]
        elif state == "NY":
            pref_cat = rng.choices(_PRODUCT_CATEGORIES, weights=[0.10, 0.45, 0.15, 0.15, 0.15])[0]
        else:
            pref_cat = rng.choice(_PRODUCT_CATEGORIES)

        category_products = [p for p in products if p[2] == pref_cat]

        # Time window for this customer's orders
        active_window_end = (
            REFERENCE_DATE - timedelta(days=rng.randint(91, 180))
            if is_churned
            else REFERENCE_DATE - timedelta(days=rng.randint(0, 60))
        )
        active_window_start = max(signup, active_window_end - timedelta(days=240))
        window_days = max((active_window_end - active_window_start).days, 1)

        for _ in range(n_orders):
            order_id += 1
            # Prefer category 70% of the time, otherwise random product
            if rng.random() < 0.70 and category_products:
                product = rng.choice(category_products)
            else:
                product = rng.choice(products)
            pid, _pname, _pcat, price, _stock, _thresh = product

            qty = rng.choices([1, 1, 1, 2, 2, 3, 4, 5, 10], k=1)[0]
            amount = round(qty * price, 2)
            order_offset = rng.randint(0, window_days)
            order_date = active_window_start + timedelta(days=order_offset)
            status = rng.choices(["delivered", "shipped", "pending"], weights=[0.80, 0.13, 0.07])[0]
            rows.append((order_id, cid, pid, qty, amount, str(order_date), status))

    # Inject 3 pricing anomalies: amount doesn't match qty * catalog price.
    # These are real-world data-quality bugs (data entry error, secret discount,
    # promo period not recorded). Multi-step anomaly tasks should find these.
    anomaly_targets = [r for r in rows if r[3] == 1][:3]  # 3 single-unit orders
    for i, row in enumerate(anomaly_targets):
        oid, cid, pid, qty, _amt, odate, status = row
        catalog_price = prod_by_id[pid][3]
        if i == 0:
            anomalous_amount = round(catalog_price * 5.2, 2)  # bug: charged 5x
        elif i == 1:
            anomalous_amount = round(catalog_price * 0.18, 2)  # 82% off promo
        else:
            anomalous_amount = round(catalog_price + 0.01, 2)  # off by penny (data entry)
        idx = rows.index(row)
        rows[idx] = (oid, cid, pid, qty, anomalous_amount, odate, status)

    # Inject one orphan: customer_id that doesn't exist
    order_id += 1
    orphan_product = rng.choice(products)
    rows.append(
        (
            order_id,
            999,  # nonexistent
            orphan_product[0],
            1,
            orphan_product[3],
            str(REFERENCE_DATE - timedelta(days=30)),
            "pending",
        )
    )

    return rows


# ---------------------------------------------------------------------------
# THE MIGRATION  ("Tom moved customer attributes into a JSON column")
# ---------------------------------------------------------------------------
#
# The demo's premise: a teammate ran a migration. ``customers.state`` and
# ``customers.segment`` used to be plain columns; they were consolidated into a
# JSON-serialized ``attributes`` TEXT column (SQLite has no JSONB), and Tom
# added new fields while he was at it — ``region`` and a nested ``acquisition``
# object (``channel`` / ``campaign``). The agent's managed-variable system
# prompt still documents the OLD flat columns.
#
# Why JSON instead of a column rename: a rename is too cheap to recover from —
# one ``describe_table`` reveals the new column name and the agent is back on
# track in a single turn, so it barely moves tokens/latency. With JSON,
# ``describe_table`` only reveals ``attributes TEXT`` — it CANNOT show the keys
# inside the blob. So the agent has to SELECT rows and ``json.loads`` /
# ``json_extract`` them, often sampling several rows to see the full structure
# (including Tom's new fields), before it can rewrite its query. That's a real
# multi-step investigation on EVERY run — the "expensive moment" that makes the
# wasted work show up as meaningful tokens/latency/cost.
#
# Live ``customers`` schema: (id, name, email, attributes, signup_date), where
# ``attributes`` is JSON like:
#   {"state": "CA", "segment": "enterprise", "region": "West",
#    "acquisition": {"channel": "referral", "campaign": "q1-launch"}}
# The other tables (orders, products, meta) are documented correctly in the
# "before" prompt — the customers JSON is the sole mismatch, so the demo turns
# on one clean, believable issue.
#
# The fix the optimizer should propose: bake the discovered JSON structure into
# the prompt (the attributes keys + how to extract them) so the agent stops
# re-investigating it every run.
#
# Tuning: to make the investigation deeper/wider, add more nested fields, make
# the structure vary across rows, migrate an ``orders`` field too, or bump the
# row counts (``range(1, 51)`` in ``_build_customers``) so each sampled-rows
# query pulls more context. Inserts below stay POSITIONAL.
# ---------------------------------------------------------------------------


def seed_database(db: Database) -> None:
    """Seed ``db`` with realistic, deterministic e-commerce data.

    The schema and contents are fixed (seed=42) so eval cases are
    reproducible across runs — essential for prompt optimization.

    NOTE: the column names here are the POST-migration ("new") schema. See
    the migration note above; the demo's managed-variable prompt deliberately
    documents the pre-migration names.
    """
    rng = random.Random(_SEED)
    conn = db.conn

    conn.execute(
        """
        CREATE TABLE customers (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            email TEXT,
            attributes TEXT,
            signup_date TEXT
        )
        """
    )
    customers = _build_customers(rng)
    conn.executemany(
        "INSERT INTO customers VALUES (?, ?, ?, ?, ?)",
        [
            (cid, name, email, _customer_attributes_json(cid, state, segment), signup)
            for (cid, name, email, state, segment, signup) in customers
        ],
    )

    conn.execute(
        """
        CREATE TABLE products (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            category TEXT,
            price REAL,
            stock_on_hand INTEGER,
            reorder_threshold INTEGER
        )
        """
    )
    products = _build_products(rng)
    conn.executemany(
        "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?)",
        products,
    )

    conn.execute(
        """
        CREATE TABLE orders (
            id INTEGER PRIMARY KEY,
            customer_id INTEGER,
            product_id INTEGER,
            quantity INTEGER,
            total_amount REAL,
            placed_at TEXT,
            status TEXT
        )
        """
    )
    orders = _build_orders(rng, customers, products)
    conn.executemany(
        "INSERT INTO orders VALUES (?, ?, ?, ?, ?, ?, ?)",
        orders,
    )

    # Reference date for evaluators that need to know what "today" is.
    conn.execute(
        """
        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )
    conn.execute("INSERT INTO meta VALUES (?, ?)", ("reference_date", str(REFERENCE_DATE)))

    conn.commit()

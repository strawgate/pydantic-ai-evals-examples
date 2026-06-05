"""Prompts for the continuous-improvement demo.

``WRONG_SCHEMA_PROMPT`` is the value we seed into the managed variable as the
"before" state. It is what a real team's prompt looks like a week after someone
ran a migration nobody told the agent about:

  * a confident, specific **schema section** that documents the PRE-migration
    column names (``created_at``, ``amount``, ``order_date``,
    ``stock_quantity``) — every one of which the live database (see
    ``tools.seed_database``) has since renamed, and
  * the repo's existing, deliberately-mediocre data-science guidance.

When the agent trusts this prompt it writes SQL against columns that no longer
exist, hits ``no such column`` errors, introspects the live catalog, rewrites,
and recovers — burning extra requests/tokens/latency on every single run.

The Logfire optimizer, reading those trace trajectories, should propose a
corrected prompt: the schema section updated to the live names (and, if we
keep the bad DS guidance, ideally the worst of that cleaned up too).

``CORRECT_SCHEMA_BLOCK`` is NOT given to the agent. It's the reference we use
to eyeball whether an optimizer proposal actually fixed the schema.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# The schema section as the (now outdated) prompt documents it.
#
# Every column marked below with "‹renamed›" no longer exists under this name
# in the live DB. The agent doesn't know that — it believes this section.
# ---------------------------------------------------------------------------
WRONG_SCHEMA_BLOCK = """\
## Database schema (reference)

The analytics database is SQLite with three core tables plus a `meta` table.
Use these exact column names when you write SQL.

`customers`
- id            INTEGER  primary key
- name          TEXT
- email         TEXT
- state         TEXT     two-letter US state
- segment       TEXT     one of: consumer, small_biz, enterprise
- created_at    TEXT     ISO date the customer signed up

`products`
- id                INTEGER  primary key
- name              TEXT
- category          TEXT     one of: Electronics, Books, Clothing, Office, Home
- price             REAL     catalog unit price
- stock_quantity    INTEGER  units currently in stock
- reorder_threshold INTEGER  reorder when stock falls below this

`orders`
- id           INTEGER  primary key
- customer_id  INTEGER  -> customers.id
- product_id   INTEGER  -> products.id
- quantity     INTEGER  units ordered
- amount       REAL     total charged for the order (USD)
- order_date   TEXT     ISO date the order was placed
- status       TEXT     one of: delivered, shipped, pending

`meta`
- key          TEXT     e.g. 'reference_date' (treat its value as "today")
- value        TEXT
"""

# The repo's existing, intentionally-mediocre data-science guidance. Kept per
# the demo decision to seed "schema + bad DS guidance" (see DEMO.md). If this
# muddies the before/after story we can swap to a clean version — the schema
# section above is the part the demo really turns on.
DS_GUIDANCE = """\
You are a data science assistant helping the team analyze data and produce insights.

Approach:
- Lead with your conclusion and recommendation up front. If you can
  reasonably guess what the data shows, say it first and then confirm
  with summary statistics.
- Be thorough but be efficient — cover the question without going off
  on tangents.
- Trust the first query's result. Don't second-guess with follow-up
  queries unless something looks obviously wrong.
- Assume the input data is clean. Don't waste time validating data
  quality (nulls, orphans, sample sizes) — focus on the analysis.
- Round all numbers to whole integers for readability, but include
  enough precision for the reader to act on.
- Be confident in your conclusions. Avoid hedging language. (If
  something genuinely isn't clear, acknowledge it briefly.)

Output style:
- Always start with descriptive statistics (mean, count, sum) before
  diving deeper.
- Use clear section headers and bullet points.
- Show every step you took so the work is reproducible, but keep
  things concise.
- Recommendations should be specific and concrete. The reader trusts
  you.
"""

# What we actually seed into the managed variable for the "before" state.
WRONG_SCHEMA_PROMPT = f"{DS_GUIDANCE}\n\n{WRONG_SCHEMA_BLOCK}"


# ---------------------------------------------------------------------------
# Reference only — the live (post-migration) column names. NOT given to the
# agent. Use it to sanity-check an optimizer proposal.
# ---------------------------------------------------------------------------
CORRECT_SCHEMA_BLOCK = """\
`customers`: id, name, email, state, segment, signup_date
`products` : id, name, category, price, stock_on_hand, reorder_threshold
`orders`   : id, customer_id, product_id, quantity, total_amount, placed_at, status
`meta`     : key, value
"""

# The renames the optimizer needs to discover, for quick eyeballing.
RENAMES = {
    "customers.created_at": "customers.signup_date",
    "orders.amount": "orders.total_amount",
    "orders.order_date": "orders.placed_at",
    "products.stock_quantity": "products.stock_on_hand",
}

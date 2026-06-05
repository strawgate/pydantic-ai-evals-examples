"""Prompts for the continuous-improvement demo.

The demo's "before" state (``BEFORE_PROMPT``) is an ordinary, reasonable system
prompt: it tells the agent the data is in a SQL database and that it can use
``list_tables`` / ``describe_table`` to find its way around — but it ships **no
schema**. So on every request the agent has to discover the schema before it can
write queries. That rediscovery is the same on every run (the schema never
changes), so it's pure repeated work — a couple of extra model turns each time
that show up as wasted latency, tokens, and cost. Nothing here tells the agent
to be inefficient; it just isn't given the one stable fact that would let it
skip the lookup.

The Logfire optimizer, reading the traces, should notice that the agent does
the same schema discovery every time and **bake the schema into the prompt** so
the agent can skip it and answer directly. ``IDEAL_AFTER_PROMPT`` is what a good
proposal looks like — it's NOT given to the agent; it's the reference we use to
eyeball whether a proposal actually eliminated the rediscovery.

(The optimizer is failure-driven by default, so getting it to optimize for
*efficiency* — fewer model calls — uses the per-variable optimizer
system-prompt override; see ``OPTIMIZER_PROMPT.md`` and DEMO.md.)
"""

from __future__ import annotations

# Reasonable, neutral analysis guidance — shared by both states. The only thing
# the demo turns on is the schema-rediscovery waste, so this part is fine as-is.
_DS_GUIDANCE = """\
When you answer:
- Derive every number from the data; never assume or hardcode values.
- Use SQL to aggregate where you can; only pull rows into Python when you need
  to compute something SQL can't.
- Watch for data-quality traps (orphaned foreign keys, nulls, tiny samples) and
  call them out rather than silently including them.
- Lead with the answer and a concrete recommendation, then show the key numbers
  that support it. Keep it tight.
"""

# A realistic, light-touch pointer at the data + tools. No schema, no mandate to
# over-explore — just "the data's in SQL, here's how to look around". The agent
# still has to discover the schema each run because it isn't given one.
_DATA_INTRO = """\
The data is in a SQL database. You can use list_tables() to see what tables are
available and describe_table(name) to see a table's columns, then query() to run
SQL and answer the question.
"""

# What we seed as the "before" value (the serving prompt).
BEFORE_PROMPT = f"""\
You are a data scientist supporting the analytics team. Answer each question
with rigorous, well-founded analysis the team can act on.

{_DATA_INTRO}
{_DS_GUIDANCE}"""


# ---------------------------------------------------------------------------
# Reference only — what a good optimizer proposal looks like. NOT given to the
# agent. The live (post-migration) column names are baked in so the agent can
# skip the per-run rediscovery entirely.
# ---------------------------------------------------------------------------
_CURRENT_SCHEMA = """\
Database schema (current and stable — query it directly; there is no need to
list or describe tables first):

  customers(id, name, email, state, segment, signup_date)
  products(id, name, category, price, stock_on_hand, reorder_threshold)
  orders(id, customer_id, product_id, quantity, total_amount, placed_at, status)
  meta(key, value)   -- meta.reference_date holds the date to treat as "today"
"""

IDEAL_AFTER_PROMPT = f"""\
You are a data scientist supporting the analytics team. Answer each question
with rigorous, well-founded analysis the team can act on.

{_CURRENT_SCHEMA}
{_DS_GUIDANCE}"""

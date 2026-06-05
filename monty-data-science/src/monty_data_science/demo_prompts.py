"""Prompts for the continuous-improvement demo (JSON-migration scenario).

The agent's prompt bakes in the database schema so it doesn't have to list /
describe tables on every run. Then a migration happened: ``customers.state`` and
``customers.segment`` were consolidated into a JSON ``attributes`` column (with
new fields ``region`` and a nested ``acquisition`` object added too — see
``tools.seed_database``). Nobody updated the prompt.

``BEFORE_PROMPT`` is that now-stale prompt: it documents ``customers`` with flat
``state`` / ``segment`` columns. So the agent writes SQL against columns that no
longer exist, gets ``no such column``, and — because ``describe_table`` only
reveals ``attributes TEXT``, not its keys — has to SELECT rows and
``json.loads`` / ``json_extract`` them, sampling a few to see the full
structure, before it can rewrite. A real multi-step investigation, every run.

``IDEAL_AFTER_PROMPT`` is what a good optimizer proposal looks like: the
``attributes`` JSON structure baked in (keys + how to extract them), so the
agent queries it directly and skips the investigation. NOT given to the agent;
it's the reference we compare proposals against.

Only ``customers`` is mis-documented — orders / products / meta are correct in
both — so the demo turns on one clean, believable issue.
"""

from __future__ import annotations

# Correct in both states — only customers differs.
_OTHER_TABLES = """\
products(id, name, category, price, stock_on_hand, reorder_threshold)
orders(id, customer_id, product_id, quantity, total_amount, placed_at, status)
meta(key, value)   -- meta.reference_date holds the date to treat as "today"\
"""

_DS_GUIDANCE = """\
When you answer:
- Derive every number from the data; never assume or hardcode values.
- Use SQL to aggregate where you can; only pull rows into Python when you need
  to compute something SQL can't.
- Watch for data-quality traps (orphaned foreign keys, nulls, tiny samples) and
  call them out rather than silently including them.
- Lead with the answer and a concrete recommendation, then show the key numbers
  that support it. Keep it tight.\
"""

# --- BEFORE: stale schema (customers still documented with flat columns) -----
_BEFORE_SCHEMA = f"""\
Database schema — query it directly; you do not need to list or describe tables.

customers(id, name, email, state, segment, signup_date)
{_OTHER_TABLES}\
"""

BEFORE_PROMPT = f"""\
You are a data scientist supporting the analytics team. Answer each question
with rigorous, well-founded analysis the team can act on.

{_BEFORE_SCHEMA}

{_DS_GUIDANCE}"""


# --- IDEAL AFTER: customers schema corrected to the JSON reality -------------
_AFTER_SCHEMA = f"""\
Database schema — query it directly; you do not need to list or describe tables.

customers(id, name, email, attributes, signup_date)
  -- `attributes` is a JSON-encoded TEXT column. Extract fields with SQLite's
  -- json_extract, e.g. json_extract(attributes, '$.state'). Keys:
  --   $.state                  (e.g. 'CA')
  --   $.segment                (consumer | small_biz | enterprise)
  --   $.region                 (West | South | Northeast | Midwest)
  --   $.acquisition.channel    (referral | organic | paid_search | partner)
  --   $.acquisition.campaign   (e.g. 'q1-launch')
{_OTHER_TABLES}\
"""

IDEAL_AFTER_PROMPT = f"""\
You are a data scientist supporting the analytics team. Answer each question
with rigorous, well-founded analysis the team can act on.

{_AFTER_SCHEMA}

{_DS_GUIDANCE}"""

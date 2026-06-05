"""Pool of data-science questions for the background traffic generator.

These mimic the steady stream of asks a data-science agent gets in production.
They are deliberately spread across the dataset so that, with the wrong-schema
prompt in place, essentially every question trips over at least one renamed
column:

  * revenue / CLV / pricing  -> orders.total_amount   (prompt says ``amount``)
  * cohort / retention / time -> customers.signup_date, orders.placed_at
                                  (prompt says ``created_at`` / ``order_date``)
  * inventory / reorder       -> products.stock_on_hand
                                  (prompt says ``stock_quantity``)

A worker picks one at random each iteration. Order is irrelevant; variety keeps
the traces realistic and gives the optimizer a broad sample of failures.
"""

from __future__ import annotations

QUESTIONS: list[str] = [
    # --- revenue / customer value (orders.total_amount) ---
    "Marketing wants a retention campaign aimed at our most valuable customers. "
    "Who are they, what spend threshold defines them, and how do they differ "
    "from everyone else? Give an answer they can act on.",
    "Finance asked for total revenue broken down by customer segment, and which "
    "segment is pulling its weight relative to how many customers it has.",
    "Which states drive the most revenue, and how concentrated is our revenue in "
    "the top few? Flag any concentration risk you see.",
    "Rank product categories by total revenue and by average order value. Where "
    "should we focus merchandising effort next quarter?",
    # --- pricing anomalies (orders.total_amount vs products.price) ---
    "Someone in finance thinks a few orders were billed at the wrong price. Check "
    "whether any orders were charged an amount that doesn't match the catalog "
    "price times quantity, and tell us which ones and how far off they are.",
    # --- inventory / reorder (products.stock_on_hand) ---
    "Ops needs a reorder priority list: which products are at or below their "
    "reorder threshold, and of those, which are selling fast enough that we'll "
    "stock out soonest? Rank them.",
    "How much revenue is at risk from products that are currently low on stock "
    "relative to their recent sales velocity?",
    # --- cohort / retention / time (customers.signup_date, orders.placed_at) ---
    "Build a retention picture: group customers by the month they signed up and "
    "show how many keep ordering in the months after. Where does retention fall "
    "off a cliff?",
    "Which customers look like they've churned — no orders in the last 90 days — "
    "and what were they worth while they were active?",
    "What's the typical gap between a customer's repeat purchases, and does it "
    "differ by segment? We're trying to time re-engagement emails.",
    "Do our customers buy differently on weekdays vs weekends? Look at order "
    "volume and average order size by day of week.",
    # --- subgroup / mixed (state + category + time) ---
    "Is there a relationship between what state a customer is in and what "
    "category they buy? If so, quantify it — don't just eyeball it.",
    "Give me a health snapshot of the business this quarter: new signups, order "
    "volume, revenue trend, and anything that looks off and worth a closer look.",
]

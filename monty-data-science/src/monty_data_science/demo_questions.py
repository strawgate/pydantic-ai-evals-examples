"""Pool of data-science questions for the background traffic generator.

A broad analytics mix, chosen so the traffic collectively exercises every part
of the migration — so the optimizer has evidence to fix each changed line:

  * revenue / value           -> orders.amount        (renamed total_amount)
  * timing / cohorts / recency -> orders.order_date    (renamed placed_at)
  *                            -> customers.created_at (renamed signup_date)
  * inventory / reorder        -> products.stock_quantity (renamed stock_on_hand)
  * customer segmentation      -> customers.state / segment / region / acquisition
                                  (now inside the JSON attributes column)

A worker picks one at random each iteration.
"""

from __future__ import annotations

QUESTIONS: list[str] = [
    # --- customer segmentation (customers JSON) ---
    "Which customer segment drives the most revenue, and how does that break "
    "down by region? Tell marketing where to focus.",
    "Break our revenue down by US region. Which region should we double down on "
    "next quarter, and why?",
    "Which acquisition channel brings in the highest-value customers (by spend)? "
    "Should we shift budget between channels?",
    "Rank acquisition campaigns by the total revenue of the customers they "
    "brought in, and call out the clear winners and losers.",
    "Is there a relationship between a customer's state and what product category "
    "they buy? Quantify it rather than eyeballing it.",
    # --- timing / cohorts / recency (order_date, created_at) ---
    "Build a retention picture: group customers by the month they signed up and "
    "show how many keep ordering in the months after. Where does it fall off?",
    "Which customers look like they've churned — no orders in the last 90 days — "
    "and what were they worth while they were active?",
    "Do our customers buy differently on weekdays vs weekends? Look at order "
    "volume and average order size by day of week.",
    # --- inventory / reorder (stock_quantity) ---
    "Ops needs a reorder priority list: which products are at or below their "
    "reorder threshold, and of those, which are selling fastest? Rank them.",
    "How much revenue is at risk from products that are low on stock relative to "
    "their recent sales velocity?",
    # --- revenue / pricing (amount) ---
    "Someone in finance thinks a few orders were billed at the wrong price. Check "
    "whether any orders were charged an amount that doesn't match catalog price × "
    "quantity, and tell us which ones and how far off.",
    # --- broad ---
    "Give me a snapshot of the business this quarter: new signups, order volume, "
    "revenue trend, the segment/region mix, and anything that looks off.",
]

"""Pool of data-science questions for the background traffic generator.

These mimic the steady stream of asks a customer-analytics agent gets. They are
deliberately weighted toward the customer dimensions that the migration moved
into the JSON ``attributes`` column — segment, state, region, and acquisition
channel/campaign — so that essentially every run has to investigate that JSON
(in the "before" state) or use the baked-in structure (after).

A worker picks one at random each iteration. Each almost always needs at least
one ``attributes`` field, joined to ``orders`` for revenue/value.
"""

from __future__ import annotations

QUESTIONS: list[str] = [
    "Which customer segment drives the most revenue, and how does that break "
    "down by region? Tell marketing where to focus.",
    "Break our revenue down by US region. Which region should we double down on "
    "next quarter, and why?",
    "Which acquisition channel brings in the highest-value customers (by spend)? "
    "Should we shift budget between channels?",
    "Compare our enterprise, small_biz, and consumer segments on average customer "
    "spend and order frequency. Where's the upside?",
    "Which states have our most valuable customers? Give finance the top few and "
    "what makes them stand out.",
    "Did the customers we acquired through the 'spring-promo' campaign turn out "
    "more valuable than our organic signups? Quantify it.",
    "Marketing wants to target a high-value cohort for a campaign — identify the "
    "segment/region combinations worth the spend, with numbers.",
    "How does average customer value differ across regions and segments? Flag any "
    "combination that's punching above its weight.",
    "Which acquisition channels are underperforming on customer value and might be "
    "worth cutting? Back it up with the data.",
    "Give me a snapshot of the customer base: the mix of segments, regions, and "
    "acquisition channels, and which slices contribute the most revenue.",
    "Are enterprise customers concentrated in particular regions or channels? If "
    "so, what does that imply for how we go to market?",
    "Rank acquisition campaigns by the total revenue of the customers they "
    "brought in, and call out the clear winners and losers.",
]

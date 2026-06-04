"""Allow running as `python -m monty_data_science` — invokes the live runner."""

import asyncio

from .live import main

asyncio.run(main())

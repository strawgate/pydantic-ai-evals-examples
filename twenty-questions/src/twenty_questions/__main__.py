"""Allow running as `python -m twenty_questions`."""

import asyncio

from .live import main

asyncio.run(main())

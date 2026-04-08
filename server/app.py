from __future__ import annotations

import os

import uvicorn

from app import app as root_app

app = root_app


def main() -> None:
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "7860")))


if __name__ == "__main__":
    main()

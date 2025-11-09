from __future__ import annotations

import os
from pathlib import Path

import uvicorn


def main() -> None:
	# Ensure media dir exists relative to this module
	media_dir = Path(__file__).resolve().parent / "media"
	media_dir.mkdir(parents=True, exist_ok=True)

	# Run the FastAPI app
	reload_flag = os.environ.get("RELOAD", "0") in ("1", "true", "True")
	# uvicorn cannot use multiple workers with reload
	workers = 1 if reload_flag else int(os.environ.get("WORKERS", "1"))
	uvicorn.run(
		"naijalingo_tts.api.server:app",
		host=os.environ.get("HOST", "0.0.0.0"),
		port=int(os.environ.get("PORT", "8000")),
		reload=reload_flag,
		workers=workers,
	)


if __name__ == "__main__":
	main()



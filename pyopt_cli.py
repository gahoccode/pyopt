"""CLI entry point for the pyopt Streamlit application."""

import sys
from pathlib import Path


def main() -> None:
    """Launch the Streamlit portfolio optimization app."""
    from streamlit.web.cli import main as st_main

    app_path = str(Path(__file__).parent / "streamlit_app.py")
    sys.argv = ["streamlit", "run", app_path, "--server.headless=true"]
    st_main()


if __name__ == "__main__":
    main()

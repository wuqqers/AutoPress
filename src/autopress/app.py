"""AutoPress desktop application entry point."""
from .app_tk import APP, MODEL, PROFILE_FILE, MainWindow, main

__all__ = ["APP", "MODEL", "PROFILE_FILE", "MainWindow", "main"]

if __name__ == "__main__":
    main()

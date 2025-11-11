# Roster Generator — Local Desktop App (PySimpleGUI)

This is a simple desktop front-end for the Roster Generator built with PySimpleGUI.

How to run
1. Create and activate a virtual environment:

   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   .venv\Scripts\activate     # Windows

2. Install dependencies:

   pip install -r requirements_local.txt

3. Start the app:

   python gui.py

Notes
- PuLP is optional. If PuLP is not installed the app will fall back to the greedy assignment.
- XLSX export requires pandas + openpyxl.
- Unavailability you add via the UI is session-scoped (not written to disk). If you want persistence, you can save/modify the people CSV externally.

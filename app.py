"""Hugging Face Spaces entry point — re-executes streamlit_app.py."""
import runpy
import os

# Run streamlit_app.py as __main__ so all top-level Streamlit calls execute
runpy.run_path(os.path.join(os.path.dirname(__file__), "streamlit_app.py"), run_name="__main__")

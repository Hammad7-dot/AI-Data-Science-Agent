# docker/sandbox.Dockerfile
#
# Phase 12b - minimal sandbox image used ONLY to execute generated /
# untrusted Python scripts via tools.python_runner.run_script_in_docker.
# Do NOT run the app itself from this image -- it has no app code, no
# Streamlit, nothing beyond what a generated data-science script needs.

FROM python:3.11-slim

RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn

WORKDIR /workspace

# No ENTRYPOINT/CMD: tools.python_runner.run_script_in_docker invokes
# `python <script>.py` explicitly for each run.

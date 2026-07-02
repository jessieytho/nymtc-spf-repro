# Reproducible environment for the NYMTC SPF analysis.
# The test suite runs as the final build step, so a failing `docker build`
# means the reproducibility assertions no longer hold.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

WORKDIR /app
COPY requirements.lock.txt ./
RUN pip install --no-cache-dir -r requirements.lock.txt
COPY . .

# Build fails if the reproducibility / identification assertions fail.
RUN python tests/run_tests.py

# Default: regenerate every table and figure from source into ./outputs.
CMD ["python", "run_all.py"]

# Dockerfile for the local-dev Observatory compose. See
# docs/guides/observatory-integration.md for the adopter-facing guide.
#
# Prerequisite: the embedded Observatory SPA at
# `nanitics/observatory/ui_assets/` must exist. Run `just observatory-build`
# before `docker compose up --build` if it is missing or stale.

FROM python:3.11-slim

WORKDIR /srv

# Install the SDK with the `api` and `anthropic` extras. The repo root is
# the build context, so we copy pyproject + source in two passes to keep
# the install layer cacheable. The `nanitics/observatory/ui_assets/`
# bundle ships with the package install — no separate copy needed.
COPY pyproject.toml README.md /srv/
COPY nanitics /srv/nanitics
RUN pip install --no-cache-dir ".[api,anthropic]" uvicorn

COPY docker/observatory-dev/app.py /srv/app.py

EXPOSE 8001
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001"]

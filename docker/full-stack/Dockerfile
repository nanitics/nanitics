# Dockerfile for the full-stack Nanitics compose. See
# docker/full-stack/README.md for the one-minute bring-up.
#
# Prerequisite: the embedded Observatory SPA at
# `nanitics/observatory/ui_assets/` must exist. Run `just observatory-build`
# before `docker compose up --build` if it is missing or stale.

FROM python:3.11-slim

WORKDIR /srv

# Install the SDK with the extras the full-stack compose needs. The repo
# root is the build context, so we copy pyproject + source in two passes
# to keep the install layer cacheable. The Observatory SPA ships inside
# `nanitics/observatory/ui_assets/` — no separate copy needed.
COPY pyproject.toml README.md /srv/
COPY nanitics /srv/nanitics
RUN pip install --no-cache-dir ".[api,anthropic,openai,postgres]" uvicorn

COPY docker/full-stack/app.py /srv/app.py
COPY docker/full-stack/llm_provider.py /srv/llm_provider.py
COPY docker/full-stack/runners.py /srv/runners.py
COPY docker/full-stack/auction_routing /srv/auction_routing
COPY docker/full-stack/judge_routing /srv/judge_routing
COPY docker/full-stack/sql_analyst /srv/sql_analyst
COPY docker/full-stack/self_improver /srv/self_improver

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]

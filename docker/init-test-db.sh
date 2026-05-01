#!/bin/bash
# Creates the test database alongside the main nanitics database.
# This script is mounted as a Docker entrypoint init script and runs
# automatically when the container is first created.

set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE DATABASE nanitics_test;
EOSQL

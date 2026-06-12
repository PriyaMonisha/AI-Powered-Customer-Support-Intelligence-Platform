#!/usr/bin/env bash
# filename: docker/postgres/init-multi-db.sh
# purpose:  Create the 'airflow' metadata DB alongside POSTGRES_DB=csip on first boot.
#           Postgres only runs files in /docker-entrypoint-initdb.d/ when the data
#           directory is EMPTY (first container start with a fresh `pgdata` volume).
#           If you ever see "database airflow does not exist" after re-using an old
#           pgdata volume, run manually:
#             docker compose exec postgres psql -U postgres -c 'CREATE DATABASE airflow;'
set -euo pipefail

if [ -n "${POSTGRES_MULTIPLE_DATABASES:-}" ]; then
  for db in $(echo "$POSTGRES_MULTIPLE_DATABASES" | tr ',' ' '); do
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" <<-EOSQL
      SELECT 'CREATE DATABASE "$db"'
      WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '$db')\gexec
EOSQL
  done
fi

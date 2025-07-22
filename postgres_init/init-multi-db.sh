#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- hapa_db가 존재하지 않으면 생성 (DB-Module용)
    SELECT 'CREATE DATABASE hapa_db'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'hapa_db')\gexec

    -- hapa_development가 존재하지 않으면 생성 (Backend용)
    SELECT 'CREATE DATABASE hapa_development'
    WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = 'hapa_development')\gexec
EOSQL
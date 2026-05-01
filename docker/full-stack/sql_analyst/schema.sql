-- Self-healing SQL analyst — schema DDL, sandbox role, and seed data.
--
-- Applied by ``bootstrap.ensure_analyst_schema`` under the app's
-- privileged pool at runner registration time. Idempotent: safe to
-- re-apply on every startup. The whole file runs in a single
-- transaction — a failure anywhere rolls back and startup fails loud.
--
-- Placeholders substituted by the bootstrap helper before execution:
--   {{sandbox_role}}     — the sandbox Postgres role (login name)
--   {{sandbox_password}} — the sandbox role's password
--
-- Seed values are deterministic (no NOW(), no random()) so the
-- canonical answers in ``questions.py`` stay stable across rebuilds.

-- ── Tables ────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS regions (
    id            SERIAL PRIMARY KEY,
    name          TEXT   NOT NULL UNIQUE,
    country_code  CHAR(2) NOT NULL
);

CREATE TABLE IF NOT EXISTS customers (
    id            SERIAL PRIMARY KEY,
    name          TEXT   NOT NULL,
    email         TEXT   NOT NULL UNIQUE,
    region_id     INT    NOT NULL REFERENCES regions(id) ON DELETE RESTRICT,
    signup_date   DATE   NOT NULL
);

CREATE TABLE IF NOT EXISTS products (
    id            SERIAL PRIMARY KEY,
    sku           TEXT   NOT NULL UNIQUE,
    name          TEXT   NOT NULL,
    category      TEXT   NOT NULL,
    unit_price    NUMERIC(10,2) NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    id            SERIAL PRIMARY KEY,
    customer_id   INT    NOT NULL REFERENCES customers(id) ON DELETE RESTRICT,
    order_date    DATE   NOT NULL,
    status        TEXT   NOT NULL CHECK (status IN ('pending','shipped','delivered','cancelled'))
);

CREATE TABLE IF NOT EXISTS order_items (
    id                    SERIAL PRIMARY KEY,
    order_id              INT    NOT NULL REFERENCES orders(id) ON DELETE RESTRICT,
    product_id            INT    NOT NULL REFERENCES products(id) ON DELETE RESTRICT,
    quantity              INT    NOT NULL CHECK (quantity > 0),
    unit_price_at_order   NUMERIC(10,2) NOT NULL
);

-- ── Seed data ─────────────────────────────────────────────
-- All seeded rows use explicit id values plus ON CONFLICT DO NOTHING
-- for row-level idempotence. After seeding the explicit ids, the
-- sequences are advanced to the next unused value so later inserts
-- (if any) do not collide.

-- regions: 5 rows, ids 1..5
INSERT INTO regions (id, name, country_code) VALUES
    (1, 'North America', 'US'),
    (2, 'Europe',        'DE'),
    (3, 'Asia Pacific',  'JP'),
    (4, 'Latin America', 'BR'),
    (5, 'Middle East',   'AE')
ON CONFLICT (id) DO NOTHING;
SELECT setval(pg_get_serial_sequence('regions', 'id'), GREATEST((SELECT MAX(id) FROM regions), 1));

-- customers: 50 rows. region_id = ((id - 1) % 5) + 1 → 10 customers per region.
-- signup_date = DATE '2023-01-01' + (id - 1) days, so ids 1..50 span 2023-01-01..2023-02-19.
INSERT INTO customers (id, name, email, region_id, signup_date)
SELECT
    n                                           AS id,
    'Customer ' || LPAD(n::text, 3, '0')        AS name,
    'customer' || LPAD(n::text, 3, '0') || '@example.com' AS email,
    ((n - 1) % 5) + 1                           AS region_id,
    DATE '2023-01-01' + ((n - 1) || ' days')::interval AS signup_date
FROM generate_series(1, 50) AS n
ON CONFLICT (id) DO NOTHING;
SELECT setval(pg_get_serial_sequence('customers', 'id'), GREATEST((SELECT MAX(id) FROM customers), 1));

-- products: 30 rows across 5 categories (6 products per category).
-- category = category_names[((id - 1) % 5) + 1]
-- unit_price = 10.00 + ((id - 1) * 2.50), i.e., 10.00, 12.50, 15.00, ... 82.50
INSERT INTO products (id, sku, name, category, unit_price)
SELECT
    n                                                   AS id,
    'SKU-' || LPAD(n::text, 4, '0')                     AS sku,
    'Product ' || LPAD(n::text, 3, '0')                 AS name,
    (ARRAY['Widgets','Gadgets','Tools','Accessories','Supplies'])[((n - 1) % 5) + 1] AS category,
    (10.00 + ((n - 1) * 2.50))::NUMERIC(10,2)           AS unit_price
FROM generate_series(1, 30) AS n
ON CONFLICT (id) DO NOTHING;
SELECT setval(pg_get_serial_sequence('products', 'id'), GREATEST((SELECT MAX(id) FROM products), 1));

-- orders: 200 rows, ids 1..200. customer_id = ((id - 1) % 50) + 1.
-- order_date = DATE '2024-01-01' + ((id - 1) * 1 day) so 200 consecutive days.
-- status rotates: ids 1,5,9,...  → pending; 2,6,10,... → shipped;
-- 3,7,11,... → delivered; 4,8,12,... → cancelled.
-- That gives 50 cancelled orders total (every 4th id) which is a tidy ground truth.
INSERT INTO orders (id, customer_id, order_date, status)
SELECT
    n                                                   AS id,
    ((n - 1) % 50) + 1                                  AS customer_id,
    DATE '2024-01-01' + ((n - 1) || ' days')::interval  AS order_date,
    (ARRAY['pending','shipped','delivered','cancelled'])[((n - 1) % 4) + 1] AS status
FROM generate_series(1, 200) AS n
ON CONFLICT (id) DO NOTHING;
SELECT setval(pg_get_serial_sequence('orders', 'id'), GREATEST((SELECT MAX(id) FROM orders), 1));

-- order_items: 500 rows, ids 1..500. order_id = ((id - 1) % 200) + 1 so each
-- order gets 2–3 items. product_id = ((id - 1) % 30) + 1.
-- quantity = ((id - 1) % 5) + 1 → 1..5 rotating.
-- unit_price_at_order pinned from products.unit_price at seed time so the
-- canonical revenue-total answer is stable.
INSERT INTO order_items (id, order_id, product_id, quantity, unit_price_at_order)
SELECT
    n                                                                        AS id,
    ((n - 1) % 200) + 1                                                      AS order_id,
    ((n - 1) % 30) + 1                                                       AS product_id,
    ((n - 1) % 5) + 1                                                        AS quantity,
    (10.00 + (((n - 1) % 30) * 2.50))::NUMERIC(10,2)                         AS unit_price_at_order
FROM generate_series(1, 500) AS n
ON CONFLICT (id) DO NOTHING;
SELECT setval(pg_get_serial_sequence('order_items', 'id'), GREATEST((SELECT MAX(id) FROM order_items), 1));

-- ── Sandbox role ──────────────────────────────────────────

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = '{{sandbox_role}}') THEN
        CREATE ROLE {{sandbox_role}} LOGIN PASSWORD '{{sandbox_password}}';
    END IF;
END
$$;

-- Pin the password on every bootstrap so rotating
-- ``NANITICS_SQL_ANALYST_SANDBOX_PASSWORD`` in the env is picked up
-- on the next app start without manual intervention. Re-running this
-- with the same password is a no-op.
ALTER ROLE {{sandbox_role}} WITH LOGIN PASSWORD '{{sandbox_password}}';

-- Server-side time budget: any SELECT under this role that exceeds
-- 2s raises QueryCanceledError. Belt-and-braces with the in-tool
-- statement_timeout in ``tool.py``.
ALTER ROLE {{sandbox_role}} SET statement_timeout TO '2s';

GRANT USAGE ON SCHEMA public TO {{sandbox_role}};

-- Read-only grants on the five analyst tables only. No grants on
-- ``trace_events`` / ``runs`` — the SDK's own trace tables are not
-- visible to the sandbox role.
GRANT SELECT ON customers, products, orders, order_items, regions TO {{sandbox_role}};

-- Explicit revocations on write privileges — belt-and-braces in case
-- a future ``GRANT ALL`` slipped in upstream. These are no-ops when
-- the grant never existed.
REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON customers  FROM {{sandbox_role}};
REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON products   FROM {{sandbox_role}};
REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON orders     FROM {{sandbox_role}};
REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON order_items FROM {{sandbox_role}};
REVOKE INSERT, UPDATE, DELETE, TRUNCATE, REFERENCES, TRIGGER ON regions    FROM {{sandbox_role}};
REVOKE CREATE ON SCHEMA public FROM {{sandbox_role}};

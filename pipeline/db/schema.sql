CREATE TABLE IF NOT EXISTS tasks (
    id SERIAL PRIMARY KEY,
    task_id VARCHAR(10) UNIQUE NOT NULL,
    category VARCHAR(20) NOT NULL,
    difficulty VARCHAR(10) NOT NULL,
    title TEXT NOT NULL,
    test_count INTEGER NOT NULL,
    tags TEXT[],
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS models (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50) UNIQUE NOT NULL,
    model_id VARCHAR(100) NOT NULL,
    provider VARCHAR(30) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS runs (
    id SERIAL PRIMARY KEY,
    task_id INTEGER REFERENCES tasks(id) NOT NULL,
    model_id INTEGER REFERENCES models(id) NOT NULL,
    run_number INTEGER NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    raw_response TEXT,
    extracted_code TEXT,
    extraction_success BOOLEAN NOT NULL DEFAULT FALSE,
    latency_ttft_ms INTEGER,
    latency_total_ms INTEGER NOT NULL,
    tokens_input INTEGER NOT NULL DEFAULT 0,
    tokens_output INTEGER NOT NULL DEFAULT 0,
    pass_fail BOOLEAN NOT NULL DEFAULT FALSE,
    test_results JSONB,
    tests_passed INTEGER NOT NULL DEFAULT 0,
    tests_total INTEGER NOT NULL DEFAULT 0,
    eslint_warnings INTEGER DEFAULT 0,
    ts_any_count INTEGER DEFAULT 0,
    compiler_errors INTEGER DEFAULT 0,
    failure_mode VARCHAR(50),
    failure_notes TEXT,
    estimated_cost_usd NUMERIC(10, 6),
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(task_id, model_id, run_number)
);

CREATE TABLE IF NOT EXISTS human_votes (
    id SERIAL PRIMARY KEY,
    task_id INTEGER REFERENCES tasks(id) NOT NULL,
    model_a INTEGER REFERENCES models(id) NOT NULL,
    model_b INTEGER REFERENCES models(id) NOT NULL,
    winner VARCHAR(10) NOT NULL,
    evaluator_id VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS performance_profiles (
    id SERIAL PRIMARY KEY,
    model_id INTEGER REFERENCES models(id) NOT NULL,
    category VARCHAR(20) NOT NULL,
    pass_rate NUMERIC(5, 2),
    avg_latency_ms INTEGER,
    avg_tokens_output INTEGER,
    avg_estimated_cost NUMERIC(10, 6),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(model_id, category)
);

CREATE INDEX IF NOT EXISTS idx_runs_task ON runs(task_id);
CREATE INDEX IF NOT EXISTS idx_runs_model ON runs(model_id);
CREATE INDEX IF NOT EXISTS idx_runs_pass ON runs(pass_fail);

INSERT INTO models (name, model_id, provider) VALUES
    ('Llama-3.3-70B', 'groq/llama-3.3-70b-versatile', 'groq'),
    ('Llama-4-Scout', 'groq/meta-llama/llama-4-scout-17b-16e-instruct', 'groq'),
    ('GPT-OSS-120B', 'groq/openai/gpt-oss-120b', 'groq'),
    ('Qwen3-32B', 'groq/qwen/qwen3-32b', 'groq'),
    ('Llama-3.1-8B', 'groq/llama-3.1-8b-instant', 'groq')
ON CONFLICT (name) DO NOTHING;

INSERT INTO models (name, model_id, provider) VALUES
    ('Llama-3.3-70B', 'groq/llama-3.3-70b-versatile', 'groq'),
    ('DeepSeek-R1-Distill-70B', 'groq/deepseek-r1-distill-llama-70b', 'groq'),
    ('Qwen-QwQ-32B', 'groq/qwen-qwq-32b', 'groq'),
    ('Llama-4-Scout', 'groq/meta-llama/llama-4-scout-17b-16e-instruct', 'groq'),
    ('Mistral-Saba-24B', 'groq/mistral-saba-24b', 'groq')
ON CONFLICT (name) DO NOTHING;

INSERT INTO models (name, model_id, provider) VALUES
    ('Nemotron-3-Super', 'openrouter/nvidia/nemotron-3-super-120b-a12b:free', 'openrouter'),
    ('GLM-4.5-Air', 'openrouter/z-ai/glm-4.5-air:free', 'openrouter'),
    ('GPT-OSS-120B', 'openrouter/openai/gpt-oss-120b:free', 'openrouter'),
    ('MiniMax-M2.5', 'openrouter/minimax/minimax-m2.5:free', 'openrouter'),
    ('Qwen3-Coder-480B', 'openrouter/qwen/qwen3-coder:free', 'openrouter')
ON CONFLICT (name) DO NOTHING;

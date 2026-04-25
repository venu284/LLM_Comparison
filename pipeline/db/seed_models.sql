INSERT INTO models (name, model_id, provider) VALUES
    ('Nemotron-3-Super', 'openrouter/nvidia/nemotron-3-super-120b-a12b:free', 'openrouter'),
    ('GLM-4.5-Air', 'openrouter/z-ai/glm-4.5-air:free', 'openrouter'),
    ('GPT-OSS-120B', 'openrouter/openai/gpt-oss-120b:free', 'openrouter'),
    ('MiniMax-M2.5', 'openrouter/minimax/minimax-m2.5:free', 'openrouter'),
    ('Gemma-4-31B', 'openrouter/google/gemma-4-31b-it:free', 'openrouter')
ON CONFLICT (name) DO NOTHING;

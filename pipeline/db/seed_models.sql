INSERT INTO models (name, model_id, provider) VALUES
    ('DeepSeek-R1-0528', 'deepseek/deepseek-reasoner', 'deepseek'),
    ('Qwen3-Coder', 'dashscope/qwen3-coder', 'dashscope'),
    ('GLM-4', 'zhipuai/glm-4', 'zhipuai'),
    ('Gemini-2.5-Flash', 'gemini/gemini-2.5-flash', 'google'),
    ('Qwen2.5-Coder-32B', 'dashscope/qwen2.5-coder-32b-instruct', 'dashscope')
ON CONFLICT (name) DO NOTHING;

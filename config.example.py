# -*- coding: utf-8 -*-
"""Copy this file to config.py and fill in local paths and credentials.

The GitHub version intentionally keeps every API endpoint/key empty.
"""

FULL_LLM_POOL = {
    "deepseek-r1": {
        "model": "DeepSeek-R1",
        "source": "provider_1",
        "tag": ["671b", "reasoning"],
        "limit_token": 1000000,
    },
    "kimi-k2": {
        "model": "Kimi-K2-Instruct-0905",
        "source": "provider_2",
        "tag": ["1t", "reasoning"],
        "limit_token": 1000000,
    },
}

LLM_SETTINGS = {
    "provider_1": {"key": "", "url": "", "supports_json_mode": True},
    "provider_2": {"key": "", "url": "", "supports_json_mode": True},
    "embedding_provider": {"key": "", "url": ""},
}

SELECTED_KEYS = ["deepseek-r1", "kimi-k2"]

EMBEDDING_MODEL = {
    "model": "Qwen/Qwen3-Embedding-8B",
    "source": "embedding_provider",
    "url": "",
    "tag": ["4096d"],
    "limit_token": 1000000,
}

RUN = {
    "round_index": 1,
    "temperature": 0.01,
    "chunk_size": 2048,
    "overlap_percent": 20,
    "text_extension": 1,
    "min_cross_scores": [1],
}

DOMAINS = {
    "wildfire": {
        "pdf_dir": "./data/pdfs/wildfire",
        "markdown_dir": "./data/mds/wildfire",
        "domain_knowledge": "wildfire emission or biomass burning emission or crop residue open burning",
        "examiner_output": "./output/examiner/wildfire",
        "composite_dir": "./output/composite/wildfire",
        "composite_prefix": "Wildfire_CrossValidation",
        "embedding_dir": "./output/embeddings/wildfire",
        "crosser_dir": "./output/crosser/wildfire",
        "ensemble_dir": "./output/ensemble/wildfire",
        "questions": [1, 2, 3, 4],
    },
    "aqua": {
        "pdf_dir": "./data/pdfs/aqua",
        "markdown_dir": "./data/mds/aqua",
        "domain_knowledge": "greenhouse gas emissions from freshwater aquaculture in China",
        "examiner_output": "./output/examiner/aqua",
        "composite_dir": "./output/composite/aqua",
        "composite_prefix": "Aqua_CrossValidation",
        "embedding_dir": "./output/embeddings/aqua",
        "crosser_dir": "./output/crosser/aqua",
        "ensemble_dir": "./output/ensemble/aqua",
        "questions": [1, 2, 3],
    },
}

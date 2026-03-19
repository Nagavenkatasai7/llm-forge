"""Instruction-response pair templates for synthetic data generation."""

from __future__ import annotations

SYSTEM_PROMPTS = {
    "general": (
        "You are a helpful, accurate, and concise assistant. Answer questions "
        "based on the provided context."
    ),
    "medical": (
        "You are a medical knowledge assistant. Provide accurate, evidence-based "
        "information while noting that this is for educational purposes only."
    ),
    "legal": (
        "You are a legal knowledge assistant. Provide information based on legal "
        "documents and precedents while noting this is not legal advice."
    ),
    "code": (
        "You are a programming assistant. Provide clear, working code examples with explanations."
    ),
    "academic": (
        "You are an academic research assistant. Provide thorough, well-cited "
        "responses based on the provided literature."
    ),
}

INSTRUCTION_TEMPLATES = {
    "comprehension": [
        "Read the following passage and answer the question.\n\nPassage: {context}\n\nQuestion: What is the main idea of this passage?",
        "Based on the text below, identify the key arguments presented.\n\n{context}",
        "Explain the relationship between the concepts discussed in this passage.\n\n{context}",
    ],
    "summarization": [
        "Provide a concise summary of the following text in 2-3 sentences.\n\n{context}",
        "Write an executive summary of this document.\n\n{context}",
        "Condense the following information into bullet points.\n\n{context}",
    ],
    "extraction": [
        "Extract all named entities (people, organizations, locations) from this text.\n\n{context}",
        "Identify the key facts and figures mentioned in this passage.\n\n{context}",
        "List all technical terms defined in this text and their definitions.\n\n{context}",
    ],
    "analysis": [
        "Analyze the strengths and weaknesses of the approach described.\n\n{context}",
        "Compare and contrast the different viewpoints presented.\n\n{context}",
        "Evaluate the evidence provided and assess its reliability.\n\n{context}",
    ],
    "generation": [
        "Based on the information provided, write a detailed explanation suitable for a beginner.\n\n{context}",
        "Rewrite the following text in simpler language.\n\n{context}",
        "Create a set of study questions based on this material.\n\n{context}",
    ],
    "reasoning": [
        "Given the following premises, what conclusions can be drawn?\n\n{context}",
        "Identify any logical fallacies or inconsistencies in this argument.\n\n{context}",
        "If the assumptions in this text were reversed, what would change?\n\n{context}",
    ],
}

MULTI_TURN_TEMPLATES = [
    {
        "turns": [
            {
                "role": "user",
                "template": "Can you explain {topic} based on this context?\n\n{context}",
            },
            {"role": "assistant", "template": "{initial_response}"},
            {"role": "user", "template": "Can you elaborate on the most important aspect?"},
            {"role": "assistant", "template": "{elaboration}"},
        ]
    },
    {
        "turns": [
            {"role": "user", "template": "Summarize this text about {topic}.\n\n{context}"},
            {"role": "assistant", "template": "{summary}"},
            {"role": "user", "template": "What are the practical implications?"},
            {"role": "assistant", "template": "{implications}"},
            {"role": "user", "template": "How would you apply this in a real-world scenario?"},
            {"role": "assistant", "template": "{application}"},
        ]
    },
]

DOMAIN_TEMPLATES = {
    "medical": {
        "qa": [
            "What are the symptoms of {topic} according to the following clinical information?\n\n{context}",
            "Describe the treatment options for {topic} based on this evidence.\n\n{context}",
            "What are the risk factors for {topic} mentioned in the text?\n\n{context}",
        ],
        "system_prompt": SYSTEM_PROMPTS["medical"],
    },
    "legal": {
        "qa": [
            "What legal principles are relevant to {topic} based on this document?\n\n{context}",
            "Summarize the legal obligations described regarding {topic}.\n\n{context}",
            "What precedents apply to {topic} according to this text?\n\n{context}",
        ],
        "system_prompt": SYSTEM_PROMPTS["legal"],
    },
    "code": {
        "qa": [
            "Explain how this code implements {topic}.\n\n```\n{context}\n```",
            "What improvements would you suggest for this implementation of {topic}?\n\n```\n{context}\n```",
            "Write a unit test for the following {topic} implementation.\n\n```\n{context}\n```",
        ],
        "system_prompt": SYSTEM_PROMPTS["code"],
    },
}


def get_templates_for_domain(domain: str = "general") -> dict:
    """Get instruction templates for a specific domain."""
    if domain in DOMAIN_TEMPLATES:
        return DOMAIN_TEMPLATES[domain]

    return {
        "qa": [t for templates in INSTRUCTION_TEMPLATES.values() for t in templates],
        "system_prompt": SYSTEM_PROMPTS.get(domain, SYSTEM_PROMPTS["general"]),
    }


def format_template(template: str, **kwargs: str) -> str:
    """Safely format a template string, leaving unfilled placeholders intact."""
    try:
        return template.format(**kwargs)
    except KeyError:
        result = template
        for key, value in kwargs.items():
            result = result.replace(f"{{{key}}}", value)
        return result

def build_prompt(query: str, contexts: list[str]):
    combined_contexts = '\n'.join(contexts)

    prompt = f"""You are an AI assistant for question-answering tasks.
You must follow the following strict rules.

STRICT RULES:

1. ONLY use information directly stated in the context
2. Provide clear and direct answers in plain text format
3. Review the answer twice before outputing the final result and keep the answer concise
4. If information is not in the context, say "I cannot find this information in the context"

Context:
{combined_contexts}

Question: 
{query}

Answer:"""
    
    return prompt
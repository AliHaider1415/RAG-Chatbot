file_path = r'D:\Tech Data\SaaS\AI-Dietician\RAG Chatbot\Datasets\open_food_facts_cleaned.txt'
def parse_nutrition_to_block():
    with open(file_path, encoding='utf-8') as f:
        content = f.read()

    # Split by double newlines into blocks
    blocks = content.strip().split("\n\n")

    # Optionally clean whitespace in each block
    docs = [block.strip() for block in blocks if block.strip()]

    return docs
"""
Script to convert calories.csv to text format for RAG database.
Reads the CSV and creates formatted text documents for each food item.
"""

import pandas as pd
from pathlib import Path


def create_calorie_text_database(csv_path: str, output_path: str):
    """
    Convert nutrition CSV into a text file with formatted documents.
    Each food item becomes a formatted text entry.
    """
    # Read the CSV
    df = pd.read_csv(csv_path)

    documents = []

    for index, row in df.iterrows():
        # Clean up the calorie and kJ values
        cals = str(row['Cals_per100grams']).replace(' cal', '')
        kj = str(row['KJ_per100grams']).replace(' kJ', '')

        # Create rich document text for semantic search
        document_text = f"""Food: {row['FoodItem']}
Category: {row['FoodCategory']}
Nutritional Information:
- Calories: {cals} per 100g
- Energy: {kj} kJ per 100g
- Serving size reference: {row['per100grams']}

This is a {row['FoodCategory'].lower()} food item that provides {cals} calories per 100 grams."""

        documents.append(document_text)

    # Write all documents to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, doc in enumerate(documents):
            f.write(doc)
            # Add separator between documents (except for the last one)
            if i < len(documents) - 1:
                f.write('\n\n---\n\n')

    print(f"Successfully created {output_path}")
    print(f"Converted {len(documents)} food items from {csv_path}")
    return len(documents)


if __name__ == "__main__":
    # Define paths
    script_dir = Path(__file__).parent
    csv_path = script_dir.parent / "data" / "calories.csv"
    output_path = script_dir.parent / "data" / "calorie_database.txt"

    # Create the text database
    num_items = create_calorie_text_database(str(csv_path), str(output_path))
    print(f"\nOutput file location: {output_path}")
    print(f"Total items processed: {num_items}")

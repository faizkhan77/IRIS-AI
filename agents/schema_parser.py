# agents/schema_parser.py
import re
from typing import List, Dict, Any

def parse_schema_markdown(markdown_content: str) -> List[Dict[str, Any]]:
    """
    Parses the schema markdown file into a structured list of column information.
    
    Args:
        markdown_content: The string content of the table_context.md file.

    Returns:
        A list of dictionaries, where each dictionary represents a single column
        and contains its table, name, and description.
    """
    print("Parsing schema markdown...")
    structured_data = []
    
    # Regex to find each table block (from "### Table" to the next "### Table" or end of file)
    table_blocks = re.findall(r"### Table: `(.*?)`\n(.*?)(?=\n### Table:|\Z)", markdown_content, re.DOTALL)
    
    for table_name, block_content in table_blocks:
        # Regex to find each column line within the block
        # Captures: 1=column_name, 2=description
        column_lines = re.findall(r"- \*\*(.*?)\*\*.*?: (.*?)\n", block_content)
        for col_name, col_desc in column_lines:
            structured_data.append({
                "table_name": table_name.strip(),
                "column_name": col_name.strip(),
                "description": col_desc.strip()
            })
            
    print(f"Successfully parsed {len(structured_data)} columns from {len(table_blocks)} tables.")
    return structured_data

if __name__ == '__main__':
    # Example usage for testing the parser directly
    try:
        with open("context/table_context.md", "r") as f:
            content = f.read()
        
        parsed_data = parse_schema_markdown(content)
        
        print("\n--- Sample Parsed Data ---")
        for item in parsed_data[:5]:
            print(item)
        print("...")
        for item in parsed_data[-5:]:
            print(item)
        print("\nParser test successful.")

    except FileNotFoundError:
        print("\nERROR: Could not find 'context/table_context.md'. Make sure the path is correct.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
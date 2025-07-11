# agents/schema_parser.py
import re
from typing import List, Dict, Any

def parse_schema_markdown(markdown_content: str) -> List[Dict[str, Any]]:
    """
    Parses the schema markdown file into a structured list of column information.
    """
    print("Parsing schema markdown...")
    structured_data = []
    
    # Regex to find each table block
    table_blocks = re.findall(r"### Table: `(.*?)`\n(.*?)(?=\n### Table:|\Z)", markdown_content, re.DOTALL)
    
    for table_name, block_content in table_blocks:
        # Regex to find each column line within the block
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
    # Enhanced testing block
    try:
        with open("context/table_context.md", "r", encoding="utf-8") as f:
            content = f.read()
        
        parsed_data = parse_schema_markdown(content)
        
        if parsed_data:
            print("\n--- Sample Parsed Data ---")
            print("First 3 items:")
            for item in parsed_data[:3]:
                print(item)
            print("...")
            print("Last 3 items:")
            for item in parsed_data[-3:]:
                print(item)
            print(f"\nTotal items parsed: {len(parsed_data)}")
            print("\nParser test successful.")
        else:
            print("\nWARNING: Parser did not return any data. Is 'context/table_context.md' empty or formatted incorrectly?")

    except FileNotFoundError:
        print("\nERROR: Could not find 'context/table_context.md'. Make sure the file exists and the path is correct.")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
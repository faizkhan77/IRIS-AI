from langchain_core.tools import tool
from utils.mapping import FILEMAPPING
from sqlmodel import SQLModel
from typing import Type, List, Optional

@tool
def get_column_info(table_names: List[str]) -> str:
    """Get column names and types of given tables."""
    if not table_names:
        return "No table names provided."

    results = []

    for table_name in table_names:
        found = False

        for entry in FILEMAPPING:
            if entry.get("table_name") == table_name:
                found = True
                model_class: Type[SQLModel] = entry["model_class"]
                columns_info = []

                for field_name, field in model_class.model_fields.items():
                    col_type = str(field.annotation)
                    col_desc = field.description or "No description"
                    columns_info.append(f"{field_name} - {col_type} - {col_desc}")

                table_info = f"**{table_name}**\n" + "\n".join(columns_info)
                results.append(table_info)
                break

        if not found:
            results.append(f"**{table_name}** - Table not found.")

    return "\n\n".join(results)

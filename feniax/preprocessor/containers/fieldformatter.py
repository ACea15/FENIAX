import re

# Sample input string for demonstration purposes.
input_data = """
@dataclass
class MyClass:
    field1: int = dfield("int", default=0)
    field2: str = dfield("str", default="")
"""

# Regular expression pattern to match the specific format in data class fields
pattern = re.compile(r'(\w+): (\w+|\[.*?\]|".*?"|\'.*?\') = dfield\(".*?", (.*?)\)')

# Function to perform the substitution
def replace_dfield(match):
    field_name = match.group(1)
    field_type = match.group(2)
    remaining_args = match.group(3)
    return f'{field_name}: {field_type} = dfield("{field_name}", {remaining_args})'

# Perform the replacement using the sub method
output_data = re.sub(pattern, replace_dfield, input_data)

# Print result
print(output_data)

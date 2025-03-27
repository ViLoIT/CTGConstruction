import os
import json
import re
import sys

# Ensure correct usage
if len(sys.argv) < 3:
    print("Usage: python script.py <input_folder> <output_folder>")
    sys.exit(1)

# Get input and output folders from command-line arguments
input_folder = os.path.abspath(sys.argv[1])  # Get absolute path
output_folder = os.path.abspath(sys.argv[2])  # Get absolute path

# Extract the input folder's base name
input_folder_name = os.path.basename(input_folder)

# Create output subdirectories within the output folder
output_nodes_folder = os.path.join(output_folder, input_folder_name, "nodes")
output_edges_folder = os.path.join(output_folder, input_folder_name, "edges")

# Ensure output directories exist
os.makedirs(output_nodes_folder, exist_ok=True)
os.makedirs(output_edges_folder, exist_ok=True)

# Function to extract ID from node format
def extract_id(node_str):
    return node_str.split("id=")[-1].strip("]") if "id=" in node_str else node_str

# Function to convert snake_case or UPPER_CASE to camelCase
def to_camel_case(s):
    parts = s.lower().split("_")
    return parts[0] + "".join(p.capitalize() for p in parts[1:])

# Function to convert edge type to UPPER_CASE with underscores
def format_edge_type(edge_type):
    return re.sub(r'(?<!^)(?=[A-Z])', '_', edge_type).upper()

# Function to process JSON files
def process_json_file(input_filepath, relative_path):
    # Generate output file names dynamically while preserving subfolder structure
    base_name = os.path.splitext(os.path.basename(relative_path))[0]  # Remove `.json` extension
    output_nodes_path = os.path.join(output_nodes_folder, f"{base_name}.nodes.json")
    output_edges_path = os.path.join(output_edges_folder, f"{base_name}.edges.json")

    # Load JSON data
    with open(input_filepath, "r", encoding="utf-8") as f:
        try:
            cpg_data = json.load(f)
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON file: {input_filepath}")
            return

    nodes = []
    edges = []

    # Process each function in the JSON
    for function in cpg_data.get("functions", []):
        # Extract function node itself
        function_id = extract_id(function["id"])
        function_label = function["id"].split("[label=")[-1].split(";")[0]

        function_entry = {
            "id": function_id,
            "_label": function_label,
            "function": function.get("function", ""),
            "file": function.get("file", "")
        }
        nodes.append(function_entry)

        # Process AST, CFG, PDG Nodes
        for key in ["AST", "CFG", "PDG"]:  # Handling different graph types
            if key in function:
                for node in function[key]:
                    node_id = extract_id(node["id"])
                    node_label = node["id"].split("[label=")[-1].split(";")[0]

                    # Convert properties to camelCase and flatten at the top level
                    properties = {to_camel_case(prop["key"]): prop["value"] for prop in node.get("properties", [])}

                    # Create node entry
                    node_entry = {"id": node_id, "_label": node_label, **properties}
                    nodes.append(node_entry)

                    # Process edges
                    if "edges" in node:
                        for edge in node["edges"]:
                            edge_type = format_edge_type(edge["id"].split(".")[-1].split("@")[0])  # Convert type format
                            edge_entry = [
                                extract_id(edge["out"]),
                                extract_id(edge["in"]),
                                edge_type,
                                edge.get("properties", {})
                            ]
                            edges.append(edge_entry)

    # Save nodes.json
    with open(output_nodes_path, "w", encoding="utf-8") as f:
        json.dump(nodes, f, indent=2)

    # Save edges.json
    with open(output_edges_path, "w", encoding="utf-8") as f:
        json.dump(edges, f, indent=2)

    print(f"Processed: {relative_path} -> {base_name}.nodes.json & {base_name}.edges.json")

# Recursively find all JSON files in input folder
for root, _, files in os.walk(input_folder):
    for filename in files:
        if filename.endswith(".json"):
            file_path = os.path.join(root, filename)
            relative_path = os.path.relpath(file_path, input_folder)  # Get relative path for naming
            process_json_file(file_path, relative_path)

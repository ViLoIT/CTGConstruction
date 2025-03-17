import os
import json

# Input and output directories
input_folder = "./input"
output_nodes_folder = "./output/nodes"
output_edges_folder = "./output/edges"

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

# Process each JSON file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):  # Only process JSON files
        input_filepath = os.path.join(input_folder, filename)

        # Generate output file names dynamically
        base_name = os.path.splitext(filename)[0]  # Remove `.json` extension
        nodes_filepath = os.path.join(output_nodes_folder, f"{base_name}.nodes.json")
        edges_filepath = os.path.join(output_edges_folder, f"{base_name}.edges.json")

        # Load JSON data
        with open(input_filepath, "r", encoding="utf-8") as f:
            try:
                cpg_data = json.load(f)
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON file: {filename}")
                continue

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
                                edge_entry = [
                                    extract_id(edge["out"]),
                                    extract_id(edge["in"]),
                                    edge["id"].split(".")[-1].split("@")[0],  # Extract edge type
                                    # "properties": edge.get("properties", {})
                                ]
                                edges.append(edge_entry)

        # Save nodes.json
        with open(nodes_filepath, "w", encoding="utf-8") as f:
            json.dump(nodes, f, indent=2)

        # Save edges.json
        with open(edges_filepath, "w", encoding="utf-8") as f:
            json.dump(edges, f, indent=2)

        print(f"Processed: {filename} -> {base_name}.nodes.json & {base_name}.edges.json")

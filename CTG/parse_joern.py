import inspect
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
import io
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from graphviz import Digraph
from datetime import datetime
import os
import argparse
from Joern_Node import NODE
from queue import Queue

def debug(msg, noheader=False, sep="\t"):
    """Print to console with debug information."""
    caller = inspect.stack()[1]
    file_name = caller.filename
    ln = caller.lineno
    now = datetime.now()
    time = now.strftime("%m/%d/%Y - %H:%M:%S")
    if noheader:
        print("\t\x1b[94m{}\x1b[0m".format(msg), end="")
        return
    print(
        '\x1b[40m[{}] File "{}", line {}\x1b[0m\n\t\x1b[94m{}\x1b[0m'.format(
            time, file_name, ln, msg
        )
    )

def nodelabel2line(label: str):
    """Given a node label, return the line number.

    Example:
    s = "METHOD_1.0: static long main()..."
    nodelabel2line(s)
    >>> '1.0'
    """

    if pd.isna(label):  # ✅ Check if label is NaN (missing)
        return "UNKNOWN"  # Or return "" or 0 if needed
    
    try:
        return str(int(label))
    except:
        return label.split(":")[0].split("_")[-1]

def randcolor():
    """Generate random color."""

    def r():
        return random.randint(0, 255)

    return "#%02X%02X%02X" % (r(), r(), r())

def get_digraph(nodes, edges, edge_label=True):
    """Plot a digraph given nodes and edges list."""
    dot = Digraph(comment="Combined PDG")

    # ✅ Ensure "lineNumber" column is properly extracted
    nodes["_label"] = nodes["_label"].fillna("UNKNOWN")  # ✅ Replace NaN with default value
    nodes["lineNumber"] = nodes["_label"].apply(nodelabel2line)

    colormap = {"": "white"}

    # ✅ Iterate properly over DataFrame rows
    for _, row in nodes.iterrows():
        if row["lineNumber"] not in colormap:
            colormap[row["lineNumber"]] = randcolor()

    # ✅ Use `.iterrows()` for node creation
    for _, row in nodes.iterrows():
        style = {"style": "filled", "fillcolor": colormap[row["lineNumber"]]}
        dot.node(str(row["id"]), str(row["node_label"]), **style)

    # ✅ Use `.iterrows()` for edges
    for _, row in edges.iterrows():
        style = {"color": "black"}
        if row["etype"] == "CALL":
            style["style"] = "solid"
            style["color"] = "purple"
        elif row["etype"] == "AST":
            style["style"] = "solid"
            style["color"] = "black"
        elif row["etype"] == "CFG":
            style["style"] = "solid"
            style["color"] = "red"
        elif row["etype"] == "CDG":
            style["style"] = "solid"
            style["color"] = "blue"
        elif row["etype"] == "REACHING_DEF":
            style["style"] = "solid"
            style["color"] = "orange"
        elif "DDG" in row["etype"]:
            style["style"] = "dashed"
            style["color"] = "darkgreen"
        else:
            style["style"] = "solid"
            style["color"] = "black"

        style["penwidth"] = "1"
        if edge_label:
            dot.edge(str(row["outnode"]), str(row["innode"]), row["etype"], **style)
        else:
            dot.edge(str(row["outnode"]), str(row["innode"]), **style)

    return dot

def get_node_edges(edges_content, nodes_content, verbose=0):
    """Get node and edges given filepath (must run after run_joern).

    filepath = "/home/david/Documents/projects/singularity-sastvd/storage/processed/bigvul/before/53.c"
    """
    # outdir = Path(filepath).parent
    # outfile = outdir / Path(filepath).name

    #with open(edges_content, "r") as f:
    
    edges = json.loads(edges_content)
    # edges = pd.DataFrame(edges, columns=["innode", "outnode", "etype", "dataflow", "change_operation"])
    edges = pd.DataFrame(edges, columns=["innode", "outnode", "etype", "dataflow"])
    edges = edges.fillna("")
    nodes = json.loads(nodes_content)
    nodes = pd.DataFrame.from_records(nodes)
    if "controlStructureType" not in nodes.columns:
        nodes["controlStructureType"] = ""
    nodes = nodes.fillna("")
    try:
        nodes = nodes[
            # ["id", "_label", "name", "code", "lineNumber", "controlStructureType", "ALPHA"]
            ["id", "_label", "name", "code", "lineNumber", "controlStructureType"]
        ]
    except Exception as E:
        print(f"Failed: {E}")
        if verbose > 1:
            debug(f"Failed: {E}")
        return None

    # Assign line number to local variables
    #with open(filepath, "r") as f:
    #code = io.StringIO(source_content).readlines()
    #lmap = assign_line_num_to_local(nodes, edges, code)
    # nodes.lineNumber = nodes.apply(
    #     lambda x: lmap[x.id] if x.id in lmap else x.lineNumber, axis=1
    # )
    # nodes.lineNumber = nodes.apply(
    #     lambda x: lmap[x.id] if x.id in lmap else x.lineNumber, axis=1
    # )
    nodes = nodes.fillna("")

    # Assign node name to node code if code is null
    nodes.code = nodes.apply(lambda x: "" if x.code == "<empty>" else x.code, axis=1)
    nodes.code = nodes.apply(lambda x: x.code if x.code != "" else x["name"], axis=1)

    # Assign node label for printing in the graph
    nodes["node_label"] = (
            nodes._label + "_" + nodes.lineNumber.astype(str) + ": " + nodes.code
    )

    # Filter by node type
    nodes = nodes[nodes._label != "COMMENT"]
    nodes = nodes[nodes._label != "FILE"]

    # Filter by edge type
    edges = edges[edges.etype != "CONTAINS"]
    edges = edges[edges.etype != "SOURCE_FILE"]
    edges = edges[edges.etype != "DOMINATE"]
    edges = edges[edges.etype != "POST_DOMINATE"]

    # Remove nodes not connected to line number nodes (maybe not efficient)
    edges = edges.merge(
        nodes[["id", "lineNumber"]].rename(columns={"lineNumber": "line_out"}),
        left_on="outnode",
        right_on="id",
    )
    edges = edges.merge(
        nodes[["id", "lineNumber"]].rename(columns={"lineNumber": "line_in"}),
        left_on="innode",
        right_on="id",
    )
    edges = edges[(edges.line_out != "") | (edges.line_in != "")]

    # Uniquify types
    edges.outnode = edges.apply(
        lambda x: f"{x.outnode}_{x.innode}" if x.line_out == "" else x.outnode, axis=1
    )
    nodes, edges = filterInfo(nodes, edges)
    typemap = nodes[["id", "name"]].set_index("id").to_dict()["name"]

    linemap = nodes.set_index("id").to_dict()["lineNumber"]
    for e in edges.itertuples():
        if type(e.outnode) == str:
            lineNum = linemap[e.innode]
            # node_label = f"TYPE_{lineNum}: {typemap[int(e.outnode.split('_')[0])]}"
            node_id = e.outnode.split('_')[0]
            node_name = typemap.get(node_id, "UNKNOWNLONG")
            if node_name != "UNKNOWNLONG":
                node_label = f"TYPE_{lineNum}: {node_name}"
                nodes = nodes.append(
                    {"id": e.outnode, "node_label": node_label, "lineNumber": lineNum},
                    ignore_index=True,
                )
    return nodes, edges

def generate_node_content(nodes):
    """Generates node content and adds it as a new column."""
    content = []
    for i, n in nodes.iterrows():
        tmp = NODE(nodes.at[i, "id"], nodes.at[i, "_label"], nodes.at[i, "code"], nodes.at[i, "name"])
        content.append(tmp.print_node())
    nodes["node_content"] = content
    return nodes

def plot_node_edges(edges_file, nodes_file, lineNumber: int = -1, filter_edges=[]):
    """Plot node edges given filepath (must run after get_node_edges).

    TO BE DEPRECATED.
    """
    nodes, edges = get_node_edges(edges_file, nodes_file)

    if len(filter_edges) > 0:
        edges = edges[edges.etype.isin(filter_edges)]

    # Draw graph
    if lineNumber > 0:
        nodesforline = set(nodes[nodes.lineNumber == lineNumber].id.tolist())
    else:
        nodesforline = set(nodes.id.tolist())

    edges_new = edges[
        (edges.outnode.isin(nodesforline)) | (edges.innode.isin(nodesforline))
        ]
    nodes_new = nodes[
        nodes.id.isin(set(edges_new.outnode.tolist() + edges_new.innode.tolist()))
    ]
    dot = get_digraph(
        nodes_new[["id", "node_label"]].to_numpy().tolist(),
        edges_new[["outnode", "innode", "etype"]].to_numpy().tolist(),
    )
    dot.render("/tmp/tmp.gv", view=True)

def neighbour_nodes(nodes, edges, nodeids: list, hop: int = 1, intermediate=True):
    """Given nodes, edges, nodeid, return hop neighbours.

    nodes = pd.DataFrame()

    """
    nodes_new = (
        nodes.reset_index(drop=True).reset_index().rename(columns={"index": "adj"})
    )
    id2adj = pd.Series(nodes_new.adj.values, index=nodes_new.id).to_dict()
    adj2id = {v: k for k, v in id2adj.items()}

    arr = []
    for e in zip(edges.innode.map(id2adj), edges.outnode.map(id2adj)):
        arr.append([e[0], e[1]])
        arr.append([e[1], e[0]])

    arr = np.array(arr)
    shape = tuple(arr.max(axis=0)[:2] + 1)
    coo = sparse.coo_matrix((np.ones(len(arr)), (arr[:, 0], arr[:, 1])), shape=shape)

    def nodeid_neighbours_from_csr(nodeid):
        return [
            adj2id[i]
            for i in csr[
                id2adj[nodeid],
            ]
                .toarray()[0]
                .nonzero()[0]
        ]

    neighbours = defaultdict(list)
    if intermediate:
        for h in range(1, hop + 1):
            csr = coo.tocsr()
            csr **= h
            for nodeid in nodeids:
                neighbours[nodeid] += nodeid_neighbours_from_csr(nodeid)
        return neighbours
    else:
        csr = coo.tocsr()
        csr **= hop
        for nodeid in nodeids:
            neighbours[nodeid] += nodeid_neighbours_from_csr(nodeid)
        return neighbours

def rdg(edges, gtype):
    """Reduce graph given type."""
    if gtype == "reftype":
        return edges[(edges.etype == "EVAL_TYPE") | (edges.etype == "REF")]
    if gtype == "ast":
        return edges[(edges.etype == "AST")]
    if gtype == "pdg":
        return edges[(edges.etype == "REACHING_DEF") | (edges.etype == "CDG")]
    if gtype == "cfgcdg":
        return edges[(edges.etype == "CFG") | (edges.etype == "CDG")]
    if gtype == "all":
        return edges[
            (edges.etype == "REACHING_DEF")
            | (edges.etype == "CDG")
            | (edges.etype == "AST")
            | (edges.etype == "EVAL_TYPE")
            | (edges.etype == "REF")
            ]

def assign_line_num_to_local(nodes, edges, code):
    """Assign line number to local variable in CPG."""
    label_nodes = nodes[nodes._label == "LOCAL"].id.tolist()
    onehop_labels = neighbour_nodes(nodes, rdg(edges, "ast"), label_nodes, 1, False)
    twohop_labels = neighbour_nodes(nodes, rdg(edges, "reftype"), label_nodes, 2, False)
    node_types = nodes[nodes._label == "TYPE"]
    id2name = pd.Series(node_types.name.values, index=node_types.id).to_dict()
    node_blocks = nodes[
        (nodes._label == "BLOCK") | (nodes._label == "CONTROL_STRUCTURE")
        ]
    blocknode2line = pd.Series(
        node_blocks.lineNumber.values, index=node_blocks.id
    ).to_dict()
    local_vars = dict()
    local_vars_block = dict()
    for k, v in twohop_labels.items():
        types = [i for i in v if i in id2name and i < 1000]
        if len(types) == 0:
            continue
        assert len(types) == 1, "Incorrect Type Assumption."
        block = onehop_labels[k]
        assert len(block) == 1, "Incorrect block Assumption."
        block = block[0]
        local_vars[k] = id2name[types[0]]
        local_vars_block[k] = blocknode2line[block]
    nodes["local_type"] = nodes.id.map(local_vars)
    nodes["local_block"] = nodes.id.map(local_vars_block)
    local_line_map = dict()
    for row in nodes.dropna().itertuples():
        localstr = "".join((row.local_type + row.name).split()) + ";"
        try:
            ln = ["".join(i.split()) for i in code][int(row.local_block) :].index(
                localstr
            )
            rel_ln = row.local_block + ln + 1
            local_line_map[row.id] = rel_ln
        except:
            continue
    return local_line_map

def drop_lone_nodes(nodes, edges):
    """Remove nodes with no edge connections.

    Args:
        nodes (pd.DataFrame): columns are id, node_label
        edges (pd.DataFrame): columns are outnode, innode, etype
    """
    nodes = nodes[(nodes.id.isin(edges.innode)) | (nodes.id.isin(edges.outnode))]
    return nodes

def plot_graph_node_edge_df(
        nodes, edges, nodeids=[], hop=1, drop_lone_nodes=True, edge_label=True
):
    """Plot graph from node and edge dataframes.

    Args:
        nodes (pd.DataFrame): columns are id, node_label
        edges (pd.DataFrame): columns are outnode, innode, etype
        drop_lone_nodes (bool): hide nodes with no in/out edges.
        lineNumber (int): Plot subgraph around this node.
    """
    # Drop lone nodes
    if drop_lone_nodes:
        nodes = nodes[(nodes.id.isin(edges.innode)) | (nodes.id.isin(edges.outnode))]

    # Get subgraph
    if len(nodeids) > 0:
        nodeids = nodes[nodes.lineNumber.isin(nodeids)].id
        keep_nodes = neighbour_nodes(nodes, edges, nodeids, hop)
        keep_nodes = set(list(nodeids) + [i for j in keep_nodes.values() for i in j])
        nodes = nodes[nodes.id.isin(keep_nodes)]
        edges = edges[
            (edges.innode.isin(keep_nodes)) & (edges.outnode.isin(keep_nodes))
            ]

    dot = get_digraph(
        nodes[["id", "node_label"]].to_numpy().tolist(),
        edges[["outnode", "innode", "etype"]].to_numpy().tolist(),
        edge_label=edge_label,
    )
    dot.render("/tmp/tmp.gv", view=True)

# Function to safely load JSON files
def load_json_file(filepath):
    """
    Safely loads a JSON file and returns its content.
    Handles missing files, empty files, and JSON parsing errors.

    :param filepath: Path to the JSON file.
    :return: JSON string (str) or None if an error occurs.
    """
    if not os.path.exists(filepath):
        print(f"❌ Error: File not found - {filepath}")
        return None

    if os.path.getsize(filepath) == 0:
        print(f"⚠️ Warning: Empty file - {filepath}")
        return None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)  # Load as a Python object
            return json.dumps(data)  # Convert back to JSON string
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON format in {filepath} - {e}")
        return None

def find_root_node(stmt_edges):
    outnodes = stmt_edges["outnode"].tolist()
    innodes = stmt_edges["innode"].tolist()
    for n in outnodes:
        if n not in innodes:
            return n

def aggregate_edges(stmt_nodes, edges):
    node_ids = stmt_nodes["id"].to_list()
    stmt_edges = edges[(edges["innode"].isin(node_ids) & edges["outnode"].isin(node_ids)) & (edges["etype"] == "AST")]
    root = find_root_node(stmt_edges)
    if root is not None:
        for n in node_ids:
            if n != root:
                edges.loc[(edges["innode"] == n) & (edges["etype"] != "AST"), "innode"] = root
                edges.loc[(edges["outnode"] == n) & (edges["etype"] != "AST"), "outnode"] = root
    return edges

def forward_slice_graph(nodes, edges, etype):
    changed_nodes = nodes["id"].tolist()
    nodes_in_changed_edges = edges[(edges["etype"] == etype)][
        "outnode"].tolist()
    for n in nodes_in_changed_edges:
        if n not in changed_nodes:
            changed_nodes.append(n)
    visited = []
    q = Queue()
    for n in changed_nodes:
        q.put(n)
    while not q.empty():
        n_id = q.get()
        visited.append(n_id)
        neighbors = edges[(edges["outnode"] == n_id) & (edges["etype"] == etype)]["innode"].to_list()
        for n in neighbors:
            if n not in visited:
                q.put(n)
    return visited

def backward_slice_graph(nodes, edges, etype):
    changed_nodes = nodes["id"].tolist()
    nodes_in_changed_edges = edges[(edges["etype"] == etype)][
        "innode"].tolist()
    for n in nodes_in_changed_edges:
        if n not in changed_nodes:
            changed_nodes.append(n)
    visited = []
    q = Queue()
    for n in changed_nodes:
        q.put(n)
    while not q.empty():
        n_id = q.get()
        visited.append(n_id)
        neighbors = edges[(edges["innode"] == n_id) & (edges["etype"] == etype)]["outnode"].to_list()
        for n in neighbors:
            if n not in visited:
                q.put(n)
    return visited

def filterInfo(nodes, edges):
    removed_nodes = nodes[nodes["lineNumber"] == ""]["id"].tolist()
    nodes = nodes[nodes["lineNumber"] != ""]
    edges = edges[~edges["innode"].isin(removed_nodes)]
    edges = edges[~edges["outnode"].isin(removed_nodes)]

    removed_nodes = nodes[nodes["_label"] == "BLOCK"]["id"].tolist()
    nodes = nodes[nodes["_label"] != "BLOCK"]
    edges = edges[~edges["innode"].isin(removed_nodes)]
    edges = edges[~edges["outnode"].isin(removed_nodes)]

    # lineNumbers = nodes.lineNumber.unique()

    # for x in lineNumbers:
    #     stmt_nodes = nodes[nodes["lineNumber"] == x]
    #     edges = aggregate_edges(stmt_nodes, edges)
    # edges = edges[edges.innode != edges.outnode]
    edges = edges.drop_duplicates(subset=["innode", "outnode", "etype"], keep='first')
    nodes = nodes.drop_duplicates(subset=["id", 'lineNumber'], keep='first')

    return nodes, edges

def trim(nodes, edges):
    nodes, edges = filterInfo(nodes, edges)
    
    kept_nodes = set()

    visited = forward_slice_graph(nodes, edges, "CDG")
    kept_nodes.update(visited)
    visited = backward_slice_graph(nodes, edges, "CDG")
    kept_nodes.update(visited)

    visited = forward_slice_graph(nodes, edges, "DDG")
    kept_nodes.update(visited)
    visited = backward_slice_graph(nodes, edges, "DDG")
    kept_nodes.update(visited)

    kept_nodes = list(kept_nodes)
    kept_line_numbers = nodes[nodes["id"].isin(kept_nodes)]["lineNumber"].to_list()
    nodes = nodes[nodes["lineNumber"].isin(kept_line_numbers)]
    kept_nodes = nodes["id"].to_list()
    edges = edges[(edges["innode"].isin(kept_nodes)) | (edges["outnode"].isin(kept_nodes))]

    nodes = drop_lone_nodes(nodes, edges)
    # # print(len(nodes))
    nodes = generate_node_content(nodes)

    node_list = nodes["id"].tolist()

    edges = edges[edges["outnode"].isin(node_list)]
    edges = edges[edges["innode"].isin(node_list)]

    return nodes, edges

# Function to process all JSON files in input folders
def process_json_files(nodes_folder, edges_folder, output_folder):
    """Processes all node and edge JSON files, converts them to CSV, and saves them."""

    # Define output folders for nodes and edges inside the output folder
    output_nodes_folder = os.path.join(output_folder, nodes_folder)
    output_edges_folder = os.path.join(output_folder, edges_folder)

    # Ensure the output directories exist
    os.makedirs(output_nodes_folder, exist_ok=True)
    os.makedirs(output_edges_folder, exist_ok=True)

    # List all node files
    node_files = [f for f in os.listdir(nodes_folder) if f.endswith(".json")]
    total_files = len(node_files) 

    for idx, node_file in enumerate(node_files, start=1):
        base_name = node_file.replace(".nodes.json", "")
        node_path = os.path.join(nodes_folder, node_file)
        edge_path = os.path.join(edges_folder, f"{base_name}.edges.json")
        
        print(f"📌 Processing file {idx}/{total_files}: {node_file}")

        # Ensure corresponding edge file exists
        if not os.path.exists(edge_path):
            print(f"⚠️ Warning: No matching edge file for {node_file}, skipping.")
            continue

        # Load JSON content
        nodes_content = load_json_file(node_path)
        edges_content = load_json_file(edge_path)

        if nodes_content is None or edges_content is None:
            print(f"❌ Error: Skipping {node_file} due to loading issues.")
            continue

        # Process data
        result = get_node_edges(edges_content, nodes_content)
        if result is None or not isinstance(result, tuple) or len(result) != 2:
            print(f"❌ Error: `get_node_edges` failed for {node_file}.")
            continue

        nodes, edges = result
        nodes, edges = trim(nodes, edges)

        if nodes is not None and edges is not None:
            print(f"✅ Successfully processed {len(nodes)} nodes and {len(edges)} edges for {base_name}.")

            # Define output file paths
            nodes_csv_path = os.path.join(output_nodes_folder, f"{base_name}_nodes.csv")
            edges_csv_path = os.path.join(output_edges_folder, f"{base_name}_edges.csv")

            # Save CSV files
            nodes.to_csv(nodes_csv_path, index=True)
            edges.to_csv(edges_csv_path, index=True)

            print(f"📂 Saved CSV:\n  {nodes_csv_path}\n  {edges_csv_path}")
        else:
            print(f"❌ Error: `get_node_edges` returned invalid data for {node_file}.")
            
# Main function to handle command-line arguments
def main():
    start_time = datetime.now()
    print(f"🚀 Processing started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    parser = argparse.ArgumentParser(description="Convert JSON files from nodes and edges folders to CSV.")
    parser.add_argument("nodes_folder", type=str, help="Path to the nodes folder.")
    parser.add_argument("edges_folder", type=str, help="Path to the edges folder.")
    parser.add_argument("output_folder", type=str, help="Path to save CSV output.")

    args = parser.parse_args()
    print(f'Nodes folder: {args.nodes_folder}')
    print(f'Edges folder: {args.edges_folder}')
    print(f'CSV folder: {args.output_folder}')
 
    process_json_files(args.nodes_folder, args.edges_folder, args.output_folder)
    end_time = datetime.now()  # Record end time
    elapsed_time = end_time - start_time  # Calculate elapsed time

    print(f"✅ Processing completed at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"⏳ Total execution time: {elapsed_time}")

# Run the script
if __name__ == "__main__":
    main()
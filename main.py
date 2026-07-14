import os
import laspy
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import kmapper as km
from gtda.homology import VietorisRipsPersistence
from gtda.graphs import GraphGeodesicDistance
import networkx as nx
from sklearn.cluster import DBSCAN
import gudhi as gd

from persim import plot_diagrams
from gtda.plotting import plot_diagram

from flooder import flood_complex, generate_landmarks

DEVICE = "cuda"
NUM_LMS = 200       # Number of landmarks for Flood complex
PERS_THRES = 0.2    # Threshold for filtering persistent features


def compute_flood_complex_ph(las):
    """
    Computes the flood complex and its persistence diagrams from the given point cloud.
    """
    pts = torch.tensor(las.xyz, device=DEVICE)
    lms = generate_landmarks(pts, NUM_LMS)

    stree = flood_complex(pts, lms, return_simplex_tree=True)
    stree.compute_persistence()
    ph = [stree.persistence_intervals_in_dimension(i) for i in range(3)]

    # Plot persistence diagrams
    plt.figure(figsize=(8, 8))
    plot_diagrams(ph, labels=['$H_0$', '$H_1$', '$H_2$'])
    plt.title("Persistence Diagram of Flood Complex")
    plt.show()


    # Visualize points contributing to the Persistence Features
    pairs = stree.persistence_pairs()

    # We want to identify the most persistent H features
    h0_labels = np.zeros(len(lms), dtype=np.uint8)
    h1_labels = np.zeros(len(lms), dtype=np.uint8)
    h2_labels = np.zeros(len(lms), dtype=np.uint8)

    landmark_masks = {0: h0_labels, 1: h1_labels, 2: h2_labels}
    for h in range(3):
        print(f"Number of H{h} features: {len(ph[h])}")
        h_landmarks = []
        for birth_simplex, death_simplex in pairs:
            # We are looking for H{h} features (born at dimension h, died at dimension h+1)
            if len(birth_simplex) == (h + 1): 
                # Calculate persistence lifetime
                birth_val = stree.filtration(birth_simplex)
                death_val = stree.filtration(death_simplex) if death_simplex else float('inf')
                persistence = death_val - birth_val
                
                # Filter for highly persistent features (ignoring the noise near the diagonal)
                if persistence > PERS_THRES:
                    for idx in birth_simplex:
                        landmark_masks[h][idx] = 1


    # --- SAVING THE LANDMARKS TO A SEPARATE LAS FILE ---
    print("Writing landmark data to a new LAS file...")

    # Create a new header for the landmark file
    lm_header = laspy.LasHeader(point_format=las.header.point_format, version=las.header.version)

    # Add our extra h0, h1, h2 fields
    for h in range(3):
        lm_header.add_extra_dim(laspy.ExtraBytesParams(name=f"h{h}", type=np.uint8, description=f"H{h} birth landmark"))

    # Initialize the landmark LAS object
    lm_las = laspy.LasData(lm_header)

    # Pull landmarks to CPU coordinates for writing
    lm_las.xyz = lms.cpu().numpy()

    # Assign the binary arrays
    for h in range(3):
        lm_las[f"h{h}"] = landmark_masks[h]

    # Save to disk
    output_path = os.path.join(os.path.dirname(__file__), "data", "dales-landmarks-persistent.las")
    lm_las.write(output_path)
    print(f"Successfully saved landmarks to {output_path}!")


def create_visuals_for_cloudcompare(las, graph):
    """
    Helper function to export the cluster-labeled LAS file and 
    the 3D Reeb graph node/edge polyline file for CloudCompare.
    """
    # --- 1. Map Points to Cluster IDs ---
    pts = las.xyz
    point_labels = np.full(len(pts), -1, dtype=np.int32)
    node_to_id = {node_key: idx for idx, node_key in enumerate(graph['nodes'].keys())}
    
    for node_key, pt_indices in graph['nodes'].items():
        cluster_id = node_to_id[node_key]
        point_labels[pt_indices] = cluster_id

    # Create a new header from the existing las object and append our dimension
    las.add_extra_dim(laspy.ExtraBytesParams(
        name="clusters",
        type=np.int32,
        description="Mapper Reeb Graph Cluster ID"
    ))
    
    las.clusters = point_labels

    # --- 2. Build the Reeb Graph Geometry Object ---
    node_centers = {}
    for node_key, pt_indices in graph['nodes'].items():
        node_centers[node_key] = np.mean(pts[pt_indices], axis=0)

    # Gather nodes with their real-world 3D coordinates
    nodes_output = []
    for node_key, center in node_centers.items():
        nid = node_to_id[node_key]
        nodes_output.append({
            "id": nid,
            "x": float(center[0]),
            "y": float(center[1]),
            "z": float(center[2])
        })
        
    # Gather edge linkages 
    edges_output = []
    for u, links in graph['links'].items():
        for v in links:
            # Keep edges unique by tracking lower-to-higher index IDs
            if node_to_id[u] < node_to_id[v]:
                edges_output.append({
                    "source": node_to_id[u],
                    "target": node_to_id[v]
                })

    # Wrap the geometric graph layout into an explicit dictionary package
    graph_data = {
        "nodes": nodes_output,
        "edges": edges_output
    }

    return las, graph_data


def visualize_mapper_open3d(las, graph_data):
    """
    Visualizes the clustered point cloud and its Reeb graph in Open3D,
    completely removing noise points that do not belong to any cluster.
    """
    pts = las.xyz
    
    # --- 1. Get Cluster Labels & Filter Noise ---
    try:
        point_labels = np.array(las.clusters)
    except AttributeError:
        print("Error: 'clusters' field not found on the las object.")
        return

    # Create a mask to only keep points that belong to a valid cluster (id >= 0)
    clustered_mask = point_labels >= 0
    filtered_pts = pts[clustered_mask]
    filtered_labels = point_labels[clustered_mask]

    # --- 2. Set Up Cluster Colors ---
    max_node_id = max(node["id"] for node in graph_data["nodes"])
    num_clusters = max_node_id + 1
    
    # Generate high-contrast colors using a matplotlib colormap
    cmap = plt.get_cmap("tab20", num_clusters)
    node_colors = {i: cmap(i)[:3] for i in range(num_clusters)}
    
    # Map colors to the filtered points based on their cluster ID
    pc_colors = np.array([node_colors[lbl] for lbl in filtered_labels])

    # Create Open3D PointCloud object with ONLY clustered points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_pts)
    pcd.colors = o3d.utility.Vector3dVector(pc_colors)

    # --- 3. Build the Reeb Graph LineSet ---
    node_id_to_idx = {node["id"]: idx for idx, node in enumerate(graph_data["nodes"])}
    node_positions = np.array([[node["x"], node["y"], node["z"]] for node in graph_data["nodes"]])
    
    lines = []
    line_colors = []
    for edge in graph_data["edges"]:
        idx_source = node_id_to_idx[edge["source"]]
        idx_target = node_id_to_idx[edge["target"]]
        lines.append([idx_source, idx_target])
        line_colors.append(node_colors[edge["source"]])

    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(node_positions)
    line_set.lines = o3d.utility.Vector2iVector(np.array(lines))
    line_set.colors = o3d.utility.Vector3dVector(np.array(line_colors))

    # --- 4. Render Nodes as Spheres ---
    graph_geometries = [line_set]
    for node in graph_data["nodes"]:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.4)  # Adjust based on your cloud scale
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color(node_colors[node["id"]])
        sphere.translate([node["x"], node["y"], node["z"]])
        graph_geometries.append(sphere)

    print(f"Launching Open3D visualization window ({len(filtered_pts)} clustered points remaining)...")
    o3d.visualization.draw_geometries([pcd] + graph_geometries, 
                                      window_name="Mapper Reeb Graph - Clustered Points Only",
                                      width=1280, height=720,
                                      left=50, top=50,
                                      mesh_show_back_face=True)


def compute_reeb_graph_ph(las, num_intervals=10, perc_overlap=0.2, dbscan_eps=0.5, dbscan_min_samples=5):
    """
    Computes the persistent homology (PH) of the Reeb graph from the given point cloud 
    constructed based on the Mapper algorithm.
    """
    mapper = km.KeplerMapper(verbose=1)

    pts = las.xyz
    # Define the scalar function (lens) to project the data
    # lens = mapper.fit_transform(pts, projection="sum")  # Using sum as a simple lens
    lens = pts[:, 2] # Using the elevation as the lens

    graph = mapper.map(lens, 
                       pts, 
                       cover=km.Cover(n_cubes=num_intervals, perc_overlap=perc_overlap),
                       clusterer=DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples, n_jobs=-1))
    
    """ Visualize the Reeb graph """
    las, graph_data = create_visuals_for_cloudcompare(las, graph)
    las.write(os.path.join(os.path.dirname(__file__), "data", "mapper-clusters.las"))
    print("Successfully exported cluster-labeled LAS file for CloudCompare.")

    visualize_mapper_open3d(las, graph_data)

    output_html = os.path.join(os.path.dirname(__file__), "data", "full_las_skeleton.html")
    mapper.visualize(
        graph,
        path_html=output_html,
        title="Mapper Reeb Graph of Point Cloud"
    )
    print(f"Interactive skeleton saved to {output_html}")

    # Convert the Kepler Mapper graph to NetworkX
    nx_graph = km.adapter.to_nx(graph)
    num_nodes = len(nx_graph.nodes)
    node_list = list(nx_graph.nodes)
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}

    # """ 
    # Compute persistent homology (PH) over the Reeb graph's structural metric space
    # """
    # # Create an unweighted adjacency matrix of the skeleton layout
    # adj_weighted = np.full((num_nodes, num_nodes), np.inf)
    # np.fill_diagonal(adj_weighted, 0)

    # # 2. Populating with actual Euclidean distance between cluster centers
    # for u, v in nx_graph.edges:
    #     idx_u, idx_v = node_to_idx[u], node_to_idx[v]
        
    #     # Calculate geometric center of both clusters in 3D space
    #     center_u = np.mean(pts[graph['nodes'][u]], axis=0)
    #     center_v = np.mean(pts[graph['nodes'][v]], axis=0)
    #     dist = np.linalg.norm(center_u - center_v)
        
    #     adj_weighted[idx_u, idx_v] = dist
    #     adj_weighted[idx_v, idx_u] = dist


    # # 4Use Giotto's geodesic distance to build a structural metric matrix.
    # # This assigns true effective distances between clusters across your network layout.
    # ggd = GraphGeodesicDistance(method="D") 
    # distance_matrix = ggd.fit_transform(adj_weighted.reshape(1, num_nodes, num_nodes))

    # # Compute Persistence over the structural distance space
    # # We can now track H0 (branches) and H1 (macro structural loops) safely
    # # Because the graph metric space essentially turns the data into a 1-dimensional structure, 
    # # the H2 (voids/cavities) will almost certainly be completely empty.
    # gp = VietorisRipsPersistence(metric="precomputed", homology_dimensions=[0, 1])
    # persistence_diagrams = gp.fit_transform(distance_matrix)

    # fig = plot_diagram(persistence_diagrams[0])
    # fig.show()

    
    """ 
    Compute Sublevel Set Persistent Homology (PH) over the Reeb graph's scalar function
    """
    # Calculate elevations
    node_elevations = {
        node: np.mean(pts[graph['nodes'][node], 2]) for node in node_list
    }

    # Build the clean GUDHI SimplexTree
    stree = gd.SimplexTree()

    # Insert non-isolated vertices
    for node, idx in node_to_idx.items():
        stree.insert([idx], filtration=node_elevations[node])

    # Insert edges
    for u, v in nx_graph.edges:
        idx_u, idx_v = node_to_idx[u], node_to_idx[v]
        edge_filtration = max(node_elevations[u], node_elevations[v])
        stree.insert([idx_u, idx_v], filtration=edge_filtration)

    # Compute and plot
    stree.initialize_filtration() 
    stree.compute_persistence()
    ph = [stree.persistence_intervals_in_dimension(i) for i in range(3)]

    # Plot persistence diagrams
    plt.figure(figsize=(8, 8))
    plot_diagrams(ph, labels=['$H_0$', '$H_1$', '$H_2$'])
    plt.title("Sublevel Set Persistence")
    plt.show()


    # --- TOPOLOGICAL FEATURE EXTRACTION & POINT LABELING ---
    
    # Initialize a label array for the original point cloud points (0 = background/noise)
    # We will give a unique integer ID to each distinct persistent cluster
    point_labels = np.zeros(len(pts), dtype=np.int32)
    
    # Extract persistence pairs (shows which simplices caused births/deaths)
    pairs = stree.persistence_pairs()
    
    persistent_cluster_id = 1
    
    for birth_simplex, death_simplex in pairs:
        # H0 features are born at dimension 0 (0-simplex / vertex)
        if len(birth_simplex) == 1:
            birth_node_idx = birth_simplex[0]
            birth_val = stree.filtration(birth_simplex)
            
            # Determine death filtration value
            if death_simplex:
                death_val = stree.filtration(death_simplex)
            else:
                # Essential features persist to infinity. Assign a high relative value
                death_val = max(node_elevations.values()) + 5.0 
                
            persistence = death_val - birth_val
            
            # If the feature is highly persistent (not noise)
            if persistence >= PERS_THRES:
                birth_node = idx_to_node[birth_node_idx]
                print(f"Persistent Cluster {persistent_cluster_id} detected! "
                      f"Birth: {birth_val:.2f}, Death: {death_val:.2f}, Persistence: {persistence:.2f}")
                
                # --- Map the feature back to the graph's nodes ---
                # A connected component is born at 'birth_node'. As elevation increases,
                # we trace all nodes connected to it that were born BEFORE the death time.
                cluster_nodes = set()
                queue = [birth_node]
                
                while queue:
                    curr_node = queue.pop(0)
                    if curr_node not in cluster_nodes:
                        # Only include nodes that exist within this cluster's filtration lifespan
                        if node_elevations[curr_node] <= death_val:
                            cluster_nodes.add(curr_node)
                            # Traverse neighbors in the NetworkX graph
                            for neighbor in nx_graph.neighbors(curr_node):
                                if neighbor not in cluster_nodes and neighbor not in queue:
                                    queue.append(neighbor)
                
                # --- Map the graph nodes back to the raw point indices ---
                # Each Kepler Mapper node holds a list of original point indices in graph['nodes'][node]
                for node in cluster_nodes:
                    point_indices = graph['nodes'][node]
                    point_labels[point_indices] = persistent_cluster_id
                
                persistent_cluster_id += 1

    # --- SAVE THE ORIGINAL POINT CLOUD WITH CLUSTER LABELS ---
    # Create a new extra dimension in the LAS file for CloudCompare
    labeled_las = laspy.LasData(las.header)
    labeled_las.xyz = las.xyz
    
    # If the file had other dimensions, copy them over
    for name in las.point_format.dimension_names:
        if name not in ['X', 'Y', 'Z']:
            labeled_las[name] = las[name]

    # Add the custom persistent cluster ID dimension
    labeled_las.add_extra_dim(laspy.ExtraBytesParams(
        name="persistent_cluster", 
        type=np.int32, 
        description="H0 pers. clusters"
    ))
    labeled_las["persistent_cluster"] = point_labels

    # Save to file
    output_path = os.path.join(os.path.dirname(__file__), "data", "persistent-clusters.las")
    labeled_las.write(output_path)
    print(f"Saved {persistent_cluster_id - 1} persistent clusters to {output_path} for CloudCompare!")



if __name__ == "__main__":
    input_file = os.path.join(os.path.dirname(__file__), "data", "dales-sample.las")
    las = laspy.read(input_file)
    print(f"Imported LAS file from {input_file} with {len(las)} points.")

    # # Compute the flood complex and its persistence diagrams
    # compute_flood_complex_ph(las)

    # Compute the Mapper-based Reeb graph and its persistence diagrams
    compute_reeb_graph_ph(las, num_intervals=20, perc_overlap=0.2, dbscan_eps=0.4, dbscan_min_samples=5)



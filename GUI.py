import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import networkx as nx
from networkx.algorithms.cuts import conductance
import matplotlib.pyplot as plt
import community
from sklearn.metrics import silhouette_score, normalized_mutual_info_score, f1_score
from networkx.algorithms.community.quality import modularity, is_partition
from networkx.algorithms.community import greedy_modularity_communities
from networkx.algorithms.community import girvan_newman, greedy_modularity_communities, louvain
import csv

class GUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Social Media Analysis Task")
        self.root.geometry("900x1200")
        self.nodes_file_path = None
        self.edges_file_path = None
        self.nodes_number_ = 0
        
        # Browse Nodes Button and Label
        self.nodes_frame = ttk.LabelFrame(self.root)
        self.nodes_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        self.nodes_button = ttk.Button(self.nodes_frame, text="Browse Nodes File", command=self.browse_nodes)
        self.nodes_button.pack(pady=(10, 5))

        self.nodes_label = ttk.Label(self.nodes_frame, text="No nodes file selected")
        self.nodes_label.pack(pady=(0, 5))

        # Browse Edges Button and Label
        self.edges_button = ttk.Button(self.nodes_frame, text="Browse Edges File", command=self.browse_edges)
        self.edges_button.pack(pady=(5, 10))

        self.edges_label = ttk.Label(self.nodes_frame, text="No edges file selected")
        self.edges_label.pack(pady=(0, 10))
        
        # Dropdown menu for graph type
        self.type_frame = ttk.LabelFrame(self.root)  
        self.type_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        self.graph_type_label = ttk.Label(self.type_frame, text="Graph Type:")
        self.graph_type_label.pack(pady=(10, 5))

        self.graph_type_var = tk.StringVar()
        self.graph_type_dropdown = ttk.Combobox(self.type_frame, textvariable=self.graph_type_var)
        self.graph_type_dropdown['values'] = ['Directed', 'Undirected']
        self.graph_type_dropdown.set('Undirected')
        self.graph_type_dropdown.pack(pady=(0, 10))


        # Dropdown menus for drawing parameters
        self.labels_label = ttk.Label(self.type_frame, text="Nodes Labels:")
        self.labels_label.pack(pady=(10, 5))

        self.label_var = tk.StringVar()
        self.label_dropdown = ttk.Combobox(self.type_frame, textvariable=self.label_var)
        self.label_dropdown.pack(pady=(0, 10))
        
        # Button to plot graph
        self.update_attr = ttk.Button(self.type_frame, text="Update filter attribute", command=self.update_new_dropdown)
        self.update_attr.pack(pady=(10, 5), padx=30)

        self.labels_label = ttk.Label(self.type_frame, text="Nodes size:")
        self.labels_label.pack(pady=(10, 5))

        self.node_size_var = tk.StringVar()
        self.node_size_dropdown = ttk.Combobox(self.type_frame, textvariable=self.node_size_var)
        self.node_size_dropdown['values'] = ['5', '10', '15'] 
        self.node_size_dropdown.set('10') 
        self.node_size_dropdown.pack(pady=(0, 10))

        self.labels_label = ttk.Label(self.type_frame, text="Nodes Shape:")
        self.labels_label.pack(pady=(10, 5))

        self.node_shape_var = tk.StringVar()
        self.node_shape_dropdown = ttk.Combobox(self.type_frame, textvariable=self.node_shape_var)
        self.node_shape_dropdown['values'] = ['o', '^', 's', '>', 'v', '<', 'd', 'p', 'h']  
        self.node_shape_dropdown.set('o') 
        self.node_shape_dropdown.pack(pady=(0, 10))

        self.labels_label = ttk.Label(self.type_frame, text="Nodes color:")
        self.labels_label.pack(pady=(10, 5))

        self.node_color_var = tk.StringVar()
        self.node_color_dropdown = ttk.Combobox(self.type_frame, textvariable=self.node_color_var)
        self.node_color_dropdown['values'] = ['skyblue', 'red', 'green']
        self.node_color_dropdown.set('skyblue')
        self.node_color_dropdown.pack(pady=(0, 10))

        self.labels_label = ttk.Label(self.type_frame, text="Edges color:")
        self.labels_label.pack(pady=(10, 5))

        self.edge_color_var = tk.StringVar()
        self.edge_color_dropdown = ttk.Combobox(self.type_frame, textvariable=self.edge_color_var)
        self.edge_color_dropdown['values'] = ['black', 'blue', 'green']
        self.edge_color_dropdown.set('black')
        self.edge_color_dropdown.pack(pady=(0, 10))

        self.labels_label = ttk.Label(self.type_frame, text="Graph Layout:")
        self.labels_label.pack(pady=(10, 5))

        self.layout_algo_var = tk.StringVar()
        self.graph_layout_dropdown = ttk.Combobox(self.type_frame, textvariable=self.layout_algo_var)
        self.graph_layout_dropdown['values'] = ['Spring', 'Circular', 'Spectral', 'Fruchterman-Reingold']  
        self.graph_layout_dropdown.set('Spring')  
        self.graph_layout_dropdown.pack(pady=(0, 10))
        
        self.graph_frame = ttk.LabelFrame(self.root)
        self.graph_frame.pack(side=tk.LEFT, fill=tk.Y, padx=30)

        # Button to plot graph
        self.plot_button = ttk.Button(self.type_frame, text="Plot Graph", command=self.plot_graph)
        self.plot_button.pack(pady=(10, 5), padx=30)

        # Button to open metrics window
        self.open_metrics_button = ttk.Button(self.type_frame, text="Open Metrics", command=self.open_metrics_window, )
        
        # Labels to Compare
        self.compare_1_frame = ttk.LabelFrame(self.root)
        self.compare_1_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        self.actual_label = ttk.Label(self.compare_1_frame, text="Girvan Newman")
        self.actual_label.pack(pady=(10, 5))
        
        self.mod_label = ttk.Label(self.compare_1_frame, text="")
        self.mod_label.pack(pady=(10, 5))
        
        self.com_label = ttk.Label(self.compare_1_frame, text="")
        self.com_label.pack(pady=(10, 5))
        self.line = ttk.LabelFrame(self.compare_1_frame)
        self.line.pack(pady=(10, 5))
        self.actual_label = ttk.Label(self.compare_1_frame, text="Louvain")
        self.actual_label.pack(pady=(10, 5))
        
        self.mod_label_louvain = ttk.Label(self.compare_1_frame, text="")
        self.mod_label_louvain.pack(pady=(10, 5))
        
        self.com_label_louvain = ttk.Label(self.compare_1_frame, text="")
        self.com_label_louvain.pack(pady=(10, 5))
        
        self.metrices_buttons = ttk.LabelFrame(self.root)
        self.metrices_buttons.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        
        self.button_run1 = tk.Button(self.metrices_buttons, text="Conductance", command=self.run_conductance)

        self.button_run2 = tk.Button(self.metrices_buttons, text="NMI", command=self.run_nmi)

        self.button_run3 = tk.Button(self.metrices_buttons, text="Coverage", command=self.coverage)

        self.button_run4 = tk.Button(self.metrices_buttons, text="F1 score", command=self.f1_score)
        
        self.button_run5 = tk.Button(self.metrices_buttons, text="Page rank", command=self.pagerank)
             
    def open_metrics_window(self):
        metrics_window = tk.Toplevel(self.root)
        metrics_window.title("Graph Metrics")
        metrics_window.geometry("300x300")

        if self.nodes_file_path and self.edges_file_path:
            try:
                # Calculate graph metrics
                num_nodes = len(self.G.nodes)
                num_edges = len(self.G.edges)
                avg_degree = sum(dict(self.G.degree()).values()) / num_nodes
                clustering_coefficient = nx.average_clustering(self.G)
                avg_shortest_path_length = nx.average_shortest_path_length(self.G)
                
                # Labels to display metrics
                num_nodes_label = ttk.Label(metrics_window, text=f"Number of nodes: {num_nodes}")
                num_nodes_label.pack(pady=5)
                num_edges_label = ttk.Label(metrics_window, text=f"Number of edges: {num_edges}")
                num_edges_label.pack(pady=5)
                avg_degree_label = ttk.Label(metrics_window, text=f"Average degree: {avg_degree}")
                avg_degree_label.pack(pady=5)
                clustering_coefficient_label = ttk.Label(metrics_window, text=f"Clustering coefficient: {clustering_coefficient}")
                clustering_coefficient_label.pack(pady=5)  
                avg_shortest_path_length_label = ttk.Label(metrics_window, text=f"Average shortest path length: {avg_shortest_path_length}")
                avg_shortest_path_length_label.pack(pady=5)  
                
            except nx.NetworkXError as e:
                num_nodes = len(self.G.nodes)
                num_edges = len(self.G.edges)
                avg_degree = sum(dict(self.G.degree()).values()) / num_nodes
                # Labels to display metrics
                num_nodes_label = ttk.Label(metrics_window, text=f"Number of nodes: {num_nodes}")
                num_nodes_label.pack(pady=5)
                num_edges_label = ttk.Label(metrics_window, text=f"Number of edges: {num_edges}")
                num_edges_label.pack(pady=5)
                avg_degree_label = ttk.Label(metrics_window, text=f"Average degree: {avg_degree}")
                avg_degree_label.pack(pady=5)
                clustering_coefficient_label = ttk.Label(metrics_window, text=f"Clustering coefficient: {clustering_coefficient}")
                clustering_coefficient_label.pack(pady=5)  
                clustering_coefficient = nx.average_clustering(self.G)
                return
        else:
            error_label = ttk.Label(metrics_window, text="Please select both nodes and edges files.")
            error_label.pack(pady=5)

    def plot_graph(self):
        if self.nodes_file_path and self.edges_file_path:
            # Read nodes and edges data
            nodes = pd.read_csv(self.nodes_file_path)
            edges = pd.read_csv(self.edges_file_path)
            self.nodes_column_names = nodes.columns
            edges_column_names = edges.columns

            if self.graph_type_dropdown.get() == 'Undirected':
                self.G = nx.Graph()
            else:
                self.G = nx.DiGraph()

            # Add nodes
            for _, node_data in nodes.iterrows():
                node_id = node_data[self.nodes_column_names[0]]  
                attributes = {column: node_data[column] for column in self.nodes_column_names[1:]}  
                attributes['label'] = node_data[self.label_dropdown.get()]  
                attributes['shape'] = self.node_shape_dropdown.get()
                self.G.add_node(node_id, **attributes)

            # Add edges
            for _, edge_data in edges.iterrows():
                self.G.add_edge(edge_data[edges_column_names[0]], edge_data[edges_column_names[1]])
            
            if len(self.graph_frame.winfo_children()) > 0:
                # Remove existing labels and dropdowns
                for widget in self.graph_frame.winfo_children():
                    widget.destroy()
                    
            # Button to open Filter centrality
            self.filter_centrality_button = ttk.Button(self.graph_frame, text="Filter centrality", command=self.filter)
            self.filter_centrality_button.pack(pady=(10, 5), padx=10)
            
            self.betwen_var = tk.StringVar()
            self.betwen = ttk.Combobox(self.graph_frame, textvariable=self.node_color_var)
            self.betwen['values'] = ['10', '11', '12']
            self.betwen.set('10')
            self.betwen.pack(pady=(10, 5), padx=10)
            
            # Button to Filter communities
            self.filter_community_button = ttk.Button(self.graph_frame, text="Filter communities using Girvan Newman", command=self.filter_community_newman)
            self.filter_community_button.pack(pady=(10, 5), padx=10)
            
            # Button to Filter communities
            self.filter_community2_button = ttk.Button(self.graph_frame, text="Filter communities using Louvain ", command=self.filter_community_louvain)
            self.filter_community2_button.pack(pady=(10, 5), padx=10)

            self.open_metrics_button.pack(pady=(10, 5), padx=10)
            self.button_run1.pack(pady=(10, 5))
            self.button_run2.pack(pady=(10, 5))
            self.button_run3.pack(pady=(10, 5))
            self.button_run4.pack(pady=(10, 5))
            self.button_run5.pack(pady=(10, 5))
            # New dropdown list
            self.new_dropdown_label = ttk.Label(self.graph_frame, text="Filter With Attributes")
            self.new_dropdown_label.pack(pady=(10, 5))

            self.new_dropdown_var = tk.StringVar()
            self.new_dropdown = ttk.Combobox(self.graph_frame, textvariable=self.new_dropdown_var)
            self.new_dropdown.pack(pady=(0, 10))
            self.update_new_dropdown()
            self.filter_centrality_button = ttk.Button(self.graph_frame, text="Apply Filter", command=self.filter_based_on_attr)
            self.filter_centrality_button.pack(pady=(10, 5), padx=10)

            # Choose layout algorithm
            layout_algorithm = self.layout_algo_var.get()
            if layout_algorithm == 'Spring':
                self.pos = nx.spring_layout(self.G)
            elif layout_algorithm == 'Circular':
                self.pos = nx.circular_layout(self.G)
            elif layout_algorithm == 'Spectral':
                self.pos = nx.spectral_layout(self.G)
            elif layout_algorithm == 'Fruchterman-Reingold':
                self.pos = nx.fruchterman_reingold_layout(self.G)
            else:
                self.pos = nx.spring_layout(self.G)  # Default to Spring layout if not recognized

            # Create a dictionary mapping node IDs to labels
            self.node_labels = {node: data['label'] for node, data in self.G.nodes(data=True)}

            # Draw graph
            plt.figure(figsize=(8, 6))
            nx.draw(self.G, self.pos, with_labels=True, labels=self.node_labels, node_shape=self.node_shape_dropdown.get(), node_size=int(self.node_size_dropdown.get()), node_color=self.node_color_dropdown.get(), edge_color=self.edge_color_dropdown.get())
            plt.title("Graph Visualization")
            plt.show()
        else:
            print("Please select both nodes and edges files.")

    def filter_based_on_attr(self):
        
        selected_gender = self.new_dropdown.get()
        filtered_nodes = [node for node, data in self.G.nodes(data=True) if data.get(self.label_dropdown.get()) == selected_gender]
        filtered_G = self.G.subgraph(filtered_nodes)

        # Update node labels for filtered nodes
        filtered_node_labels = {node: self.node_labels[node] for node in filtered_nodes}
        # Choose layout algorithm
        layout_algorithm = self.layout_algo_var.get()
        if layout_algorithm == 'Spring':
            self.pos = nx.spring_layout(self.G)
        elif layout_algorithm == 'Circular':
            self.pos = nx.circular_layout(self.G)
        elif layout_algorithm == 'Spectral':
            self.pos = nx.spectral_layout(self.G)
        elif layout_algorithm == 'Fruchterman-Reingold':
            self.pos = nx.fruchterman_reingold_layout(self.G)
        else:
            self.pos = nx.spring_layout(self.G)  # Default to Spring layout if not recognized

        # Draw the filtered graph
        plt.figure(figsize=(8, 6))
        nx.draw(filtered_G, self.pos, with_labels=True, labels=filtered_node_labels, node_shape=self.node_shape_dropdown.get(), node_size=int(self.node_size_dropdown.get()), node_color=self.node_color_dropdown.get(), edge_color=self.edge_color_dropdown.get())
        plt.title(f"Graph Visualization based on: {self.label_dropdown.get()}")
        plt.show()

    def filter(self):
        degCent = nx.degree_centrality(self.G)
        degCent_sorted = dict(sorted(degCent.items(), key=lambda item: item[1], reverse=True))
        betCent = nx.betweenness_centrality(self.G, normalized=True, endpoints=True)
        betCent_sorted = dict(sorted(betCent.items(), key=lambda item: item[1], reverse=True))
        closCent = nx.closeness_centrality(self.G)
        closCent_sorted = dict(sorted(closCent.items(), key=lambda item: item[1], reverse=True))
        color_list = []

        N_top = int(self.betwen.get())
        colors_top_10 = ['tab:orange', 'tab:blue', 'tab:green', 'lightsteelblue', 'tab:purple']
        keys_deg_top = list(degCent_sorted)[:N_top]
        keys_bet_top = list(betCent_sorted)[:N_top]
        keys_clos_top = list(closCent_sorted)[:N_top]

        # Computing intersection of top nodes for all measures
        inter_deg_bet = set(keys_deg_top) & set(keys_bet_top)
        inter_all = inter_deg_bet & set(keys_clos_top)

        # Setting up color for nodes
        for node in self.G.nodes:
            if node in inter_all:
                color_list.append(colors_top_10[2])  
            elif node in inter_deg_bet:
                color_list.append(colors_top_10[1]) 
            elif node in keys_deg_top:
                color_list.append(colors_top_10[0])  
            elif node in keys_bet_top:
                color_list.append(colors_top_10[3])
            else:
                color_list.append('lightsteelblue')
            
        layout_algorithm = self.layout_algo_var.get()
        if layout_algorithm == 'Spring':
            self.pos = nx.spring_layout(self.G)
        elif layout_algorithm == 'Circular':
            self.pos = nx.circular_layout(self.G)
        elif layout_algorithm == 'Spectral':
            self.pos = nx.spectral_layout(self.G)
        elif layout_algorithm == 'Fruchterman-Reingold':
            self.pos = nx.fruchterman_reingold_layout(self.G)
        else:
            self.pos = nx.spring_layout(self.G)  # Default to Spring layout if not recognized

        # Draw graph
        nx.draw(self.G, self.pos, with_labels=True, node_color=color_list)

        labels = ['Top 10 deg cent', 'Top 10 bet cent', 'Top 10 clos cent', 'Top 10 deg, bet, and clos cent']
        for i in range(len(labels)):
            plt.scatter([], [], label=labels[i], color=colors_top_10[i])
            
        plt.legend(loc='center')
        plt.show()
        self.centrality()

    def filter_community_newman(self):
        communities_generator = nx.community.girvan_newman(self.G)
        self.next_level_communities = next(communities_generator)
        community_colors = {}
        community_mapping = {}
        for idx, comm in enumerate(self.next_level_communities):
            for node in comm:
                community_colors[node] = idx
                community_mapping[node] = idx 
            
        num_communities = len(self.next_level_communities)
        modularity = nx.community.modularity(self.G, self.next_level_communities)
        
        # Update labels
        self.mod_label.config(text=f"Modularity Score: {modularity}")
        self.com_label.config(text=f"Number of Communities: {num_communities}")
        self.compare_1_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)
        layout_algorithm = self.layout_algo_var.get()
        if layout_algorithm == 'Spring':
            self.pos = nx.spring_layout(self.G)
        elif layout_algorithm == 'Circular':
            self.pos = nx.circular_layout(self.G)
        elif layout_algorithm == 'Spectral':
            self.pos = nx.spectral_layout(self.G)
        elif layout_algorithm == 'Fruchterman-Reingold':
            self.pos = nx.fruchterman_reingold_layout(self.G)
        else:
            self.pos = nx.spring_layout(self.G)  # Default to Spring layout if not recognized
        
        # Draw graph
        plt.figure(figsize=(8, 6))
        nx.draw(self.G, self.pos, with_labels=True, labels=self.node_labels, node_shape=self.node_shape_dropdown.get(), node_size=int(self.node_size_dropdown.get()), node_color=[community_colors.get(node, len(self.next_level_communities)) for node in self.G.nodes()], edge_color=self.edge_color_dropdown.get(), )
        plt.title("Community Structure (Girvan-Newman)")
        plt.show()
        ##################################
        fig, axes = plt.subplots(2, max(1, num_communities), figsize=(15, 6))
        for i, community in enumerate(self.next_level_communities, start=1):
            ax = axes[0, i - 1] if num_communities > 1 else axes
            subgraph = self.G.subgraph(community)
            nx.draw(subgraph, self.pos, with_labels=True, node_size=int(self.node_size_dropdown.get()),
                    node_color=[community_colors.get(node, len(self.next_level_communities)) for node in subgraph.nodes()],
                    node_shape=self.node_shape_dropdown.get(), ax=ax)
            ax.set_title(f"Girvan-Newman: Community {i}")

        plt.tight_layout()
        plt.show()

    def filter_community_louvain(self):
        communities = list(nx.community.louvain_communities(self.G))
        community_colors = {}
        for idx, comm in enumerate(communities):
            for node in comm:
                community_colors[node] = idx

        num_communities = len(communities)
        modularity = nx.community.modularity(self.G, communities)

        # Update labels
        self.mod_label_louvain.config(text=f"Modularity Score: {modularity}")
        self.com_label_louvain.config(text=f"Number of Communities: {num_communities}")

        layout_algorithm = self.layout_algo_var.get()
        if layout_algorithm == 'Spring':
            self.pos = nx.spring_layout(self.G)
        elif layout_algorithm == 'Circular':
            self.pos = nx.circular_layout(self.G)
        elif layout_algorithm == 'Spectral':
            self.pos = nx.spectral_layout(self.G)
        elif layout_algorithm == 'Fruchterman-Reingold':
            self.pos = nx.fruchterman_reingold_layout(self.G)
        else:
            self.pos = nx.spring_layout(self.G)  # Default to Spring layout if not recognized

        # Draw graph with community coloring
        plt.figure(figsize=(8, 6))
        nx.draw(self.G, self.pos, with_labels=True, labels=self.node_labels, node_shape=self.node_shape_dropdown.get(),
                node_size=int(self.node_size_dropdown.get()), node_color=[community_colors.get(node, len(communities)) for node in self.G.nodes()], cmap=plt.cm.tab20, edge_color=self.edge_color_dropdown.get())
        plt.title("Community Structure (Louvain Algorithm)")
        plt.show()
        fig, axes = plt.subplots(2, max(1, num_communities), figsize=(15, 6))
        # Plot the Louvain communities
        for i, community in enumerate(communities, start=1):
            ax = axes[0, i - 1] if num_communities > 1 else axes
            subgraph = self.G.subgraph(community)
            nx.draw(subgraph, self.pos, with_labels=True, node_size=int(self.node_size_dropdown.get()),
                    node_color=[community_colors.get(node, len(communities)) for node in subgraph.nodes()],
                    node_shape=self.node_shape_dropdown.get(), ax=ax)
            ax.set_title(f"Louvain: Community {i}")

        plt.tight_layout()
        plt.show()

    def run_conductance(self):
        try:
            # Find the communities using the greedy modularity algorithm
            communities = greedy_modularity_communities(self.G)

            # Calculate the conductance for each community
            cond=''
            for community in communities:
                if len(community) == 0:
                    # Skip empty communities
                    continue
                community_edges = self.G.subgraph(community).edges()
                complement_edges = self.G.subgraph(set(self.G.nodes()) - set(community)).edges()
                volume_community = sum(self.G[u][v].get('weight', 1) for u, v in community_edges)
                volume_complement = sum(self.G[u][v].get('weight', 1) for u, v in complement_edges)
                if volume_community == 0 or volume_complement == 0:
                    # Skip communities with no edges or complement with no edges
                    continue
                conductance_value = conductance(self.G, community)
                print(f"Community {community} has conductance {conductance_value}")
                cond+='Community {}, has conductance {}'.format(community, 2*conductance_value)+'\n'
            messagebox.showinfo("Conductance", cond)
        except:
            messagebox.showerror("Error", "Invalid input: please enter a list of edges")

    def run_nmi(self):
        try:
            partition1 = community.best_partition(self.G)
            partition2 = community.best_partition(self.G)

            # Convert the partitions into lists of cluster labels
            labels1 = [partition1[node] for node in self.G.nodes()]
            labels2 = [partition2[node] for node in self.G.nodes()]

            true_labels = [partition1.get(node) for node in self.G.nodes()]
            predicted_labels = [partition1[node] for node in self.G.nodes()]

            # Compute the NMI between the two clusterings
            nmi = normalized_mutual_info_score(labels1, labels2)
            messagebox.showinfo("NMI", "Normalized Mutual Information = {}".format(nmi))
        except:
            messagebox.showerror("Error", "Invalid input: please enter a list of edges")

    def coverage(self):
        communities = list(greedy_modularity_communities(self.G))

        # calculate the coverage
        coverage = 0
        for comm in communities:
            nodes_in_comm = set(comm)
            coverage += len(nodes_in_comm) / len(self.G.nodes)
        messagebox.showinfo("Coverage", f"coverage = {coverage}")

    def f1_score(self):
        self.node_dict = {node: i for i, node in enumerate(self.G.nodes())}  # create node dictionary
        communities = list(greedy_modularity_communities(self.G))

        # create a ground truth label vector
        ground_truth = [0] * self.G.number_of_nodes()
        for i, comm in enumerate(communities):
            for node in comm:
                ground_truth[self.node_dict[node]] = i  # convert node to integer using dictionary

        # create a predicted label vector
        predicted = [0] * self.G.number_of_nodes()
        for i, comm in enumerate(communities):
            for node in comm:
                predicted[self.node_dict[node]] = i  # convert node to integer using dictionary

        # calculate the F1 score
        f1 = f1_score(ground_truth, predicted, average='weighted')
        messagebox.showinfo("F1 score", f"f1 score= {f1}")

    def pagerank(self):
            pr = nx.pagerank(self.G)
            firstmax=max(pr, key=pr.get)
            firstvalue=max(pr.values())
            pr.pop(firstmax)
            secondmax = max(pr, key=pr.get)
            secondvalue = max(pr.values())
            messagebox.showinfo("PageRank", f"{firstmax} : {firstvalue}"+"\n"+f"{secondmax} : {secondvalue}")

    def centrality(self):
        try:
            pr = nx.pagerank(self.G)
            dc = nx.degree_centrality(self.G)
            cc = nx.closeness_centrality(self.G)
            bc = nx.betweenness_centrality(self.G)
            
            # Define the filename and headers for the CSV file
            filename = "centrality_measures.csv"
            headers = ["Node", "Degree", "Closeness", "Betweenness", "Page Rank"]

            # Write the centrality measures to the CSV file
            with open(filename, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(headers)
                for node in self.G.nodes:
                    row = [
                        node,
                        round((len(self.nodes_file_path)-1) * dc[node], 2),
                        round(cc[node], 2),
                        round((((len(self.nodes_file_path)-1)*(len(self.nodes_file_path)-2)/2)) * bc[node], 2),
                        pr[node],
                    ]
                    writer.writerow(row)
            messagebox.showinfo("Communities", f"Centrality measures saved successfully in {filename}")
        except:
            messagebox.showerror("Error", "Invalid input: please enter a list of edges")
    
    def browse_nodes(self):
            self.nodes_file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
            if self.nodes_file_path:
                print("Nodes file selected:", self.nodes_file_path)
                self.list_available_labels(self.nodes_file_path)
                self.nodes_label.config(text="Nodes file path: " + self.nodes_file_path)
            else:
                print("No nodes file selected.")
            
    def browse_edges(self):
        self.edges_file_path = filedialog.askopenfilename(filetypes=[("CSV Files", "*.csv")])
        if self.edges_file_path:
            print("Edges file selected:", self.edges_file_path)
            self.edges_label.config(text="Edges file path: " + self.edges_file_path)
        else:
            print("No edges file selected.")

    def list_available_labels(self, n):
        if n:
            nodes = pd.read_csv(n)
            nodes_column_names = nodes.columns.tolist()
            self.label_dropdown['values'] = nodes_column_names
            self.label_dropdown.set(nodes_column_names[0])

    def runWindow(self):
        # Start the main GUI loop
        self.root.mainloop()
    
    def update_new_dropdown(self, event=None):
        selected_label = self.label_dropdown.get()
        if selected_label and self.nodes_file_path:
            nodes = pd.read_csv(self.nodes_file_path)
            if selected_label in nodes.columns:
                values = nodes[selected_label].unique().tolist()
                values.sort()  # Sort the values in ascending order
                self.new_dropdown["values"] = values
                self.new_dropdown.set(values[0])
            else:
                print(f"Label '{selected_label}' not found in nodes file columns.")
        else:
            print("No label selected or nodes file not provided.")

# Usage
gui = GUI()
gui.runWindow()
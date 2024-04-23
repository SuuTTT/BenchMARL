import torch
import networkx as nx
import numpy as np
from tensordict import TensorDict
import matplotlib.pyplot as plt
from torchrl.envs.libs.vmas import VmasEnv

def create_graphs_from_ted(ted, method='threshold'):
    """
    Creates a batch of graphs based on observations contained in a TensorDict (TED),
    with different methods for connecting nodes.

    Parameters:
        ted (TensorDict): TensorDict containing episode data with observations for each agent.
        method (str): Method used to create the graph. Default is 'threshold'. Other methods are not implemented.

    Returns:
        list: A list of networkx.Graph objects for each environment.
    """
    observations = ted['agents']['observation']  # Shape: [n_env, n_agents, 21]
    num_envs = observations.shape[0]
    n_agents = observations.shape[1]
    graphs = []

    for env_index in range(num_envs):
        G = nx.Graph()
        positions = observations[env_index, :, :2].cpu().numpy()  # Extract only x, y positions
        lidar_data = observations[env_index, :, 6:21].cpu().numpy()  # Extract LiDAR data

        # Add nodes with positions and LiDAR data
        for i in range(n_agents):
            G.add_node(i, pos=positions[i], lidar=lidar_data[i])

        if method == 'threshold':
            # Add edges based on the Euclidean distance using a threshold method
            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    distance = np.linalg.norm(positions[i] - positions[j])
                    if distance < 1.0:  # Example threshold
                        G.add_edge(i, j, weight=1/distance)  # Weight can be inverse of distance
        else:
            raise NotImplementedError(f"Graph creation method '{method}' is not implemented.")

        graphs.append(G)
    return graphs

def create_graphs_from_observation(observations, method='threshold'):
    """
    Creates a batch of graphs based on observations extracted a TensorDict (TED), 
    this differs from the previous function, because the transform takes the observation
    with different methods for connecting nodes.

    Parameters:
        observation: TensorDict containing episode data with observations for each agent.
        method (str): Method used to create the graph. Default is 'threshold'. Other methods are not implemented.

    Returns:
        list: A list of networkx.Graph objects for each environment.
    """
    #observations = observation['agents']['observation']  # Shape: [n_env, n_agents, 21]
   #print(observations)
    num_envs = observations.shape[0]
    n_agents = observations.shape[1]
    graphs = []

    for env_index in range(num_envs):
        G = nx.Graph()
        positions = observations[env_index, :, :2].cpu().numpy()  # Extract only x, y positions
        lidar_data = observations[env_index, :, 6:21].cpu().numpy()  # Extract LiDAR data

        # Add nodes with positions and LiDAR data
        for i in range(n_agents):
            G.add_node(i, pos=positions[i], lidar=lidar_data[i])

        if method == 'threshold':
            # Add edges based on the Euclidean distance using a threshold method
            for i in range(n_agents):
                for j in range(i + 1, n_agents):
                    distance = np.linalg.norm(positions[i] - positions[j])
                    if distance < 1.0:  # Example threshold
                        G.add_edge(i, j, weight=1/distance)  # Weight can be inverse of distance
        else:
            raise NotImplementedError(f"Graph creation method '{method}' is not implemented.")

        graphs.append(G)
    return graphs


def draw_graphs(graphs):
    """
    Draws the batch of graphs from a TensorDict.
    
    Parameters:
        graphs_td (TensorDict): TensorDict containing graphs.
    """
   
    num_envs = len(graphs)
    fig, axs = plt.subplots(num_envs, 1, figsize=(12, 6 * num_envs))  # Adjust layout if needed

    if num_envs == 1:
        axs = [axs]  # Ensure axs is iterable if only one environment

    for env_index, G in enumerate(graphs):
        ax = axs[env_index]
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, with_labels=True, ax=ax, node_size=700, node_color='blue', font_color='white')
        ax.set_title(f'Environment {env_index + 1} - Graph Representation')

    plt.tight_layout()
    plt.show()


from matplotlib import cm
def draw_graphs_and_environments(observations, graphs):
    """
    Draws the environments and corresponding graphs for easy comparison.
    
    Parameters:
        observations (torch.Tensor): Observations tensor of shape [n_env, n_agents, 21].
        graphs (list): List of networkx graphs.
    """
    num_envs = len(graphs)
    fig, axs = plt.subplots(num_envs, 2, figsize=(12, 6 * num_envs))  # Two columns for each environment

    color_map = cm.get_cmap('viridis', observations.shape[1])  # Get a color map for agents

    for env_index, G in enumerate(graphs):
        positions = observations[env_index, :, :2].cpu().numpy()
        node_colors = [color_map(i) for i in range(observations.shape[1])]

        # Plot environment
        ax_env = axs[env_index][0]
        for agent_index, pos in enumerate(positions):
            ax_env.scatter(*pos, color=node_colors[agent_index], s=100)
            ax_env.text(*pos, f'Agent {agent_index}', horizontalalignment='center', verticalalignment='center')
        ax_env.set_title(f'Environment {env_index + 1} - Positions')
        ax_env.axis('equal')

        # Plot graph
        ax_graph = axs[env_index][1]
        pos = nx.get_node_attributes(G, 'pos')
        nx.draw(G, pos, node_color=node_colors, with_labels=True, ax=ax_graph, node_size=700)
        ax_graph.set_title(f'Environment {env_index + 1} - Graph Representation')

    plt.tight_layout()
    plt.show()

# Example environment setup and execution
# Assuming a function 'env.reset()' returns a TensorDict containing 'agents' -> 'observation' tensor
env = VmasEnv(
    scenario='discovery',
    num_envs=2,
    n_agents=5,
    n_targets=1,
    seed=0,
    device='cuda'
)


import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
import networkx as nx

import community as community_louvain

def create_clusters_louvain(observation, n_target, n_agents, agents_per_target):
    """
    Use the Louvain method for community detection on graphs constructed from observations.

    Parameters:
    - observation: A list of data representing the observations of agents, which will be converted to graphs.

    Returns:
    - list: A list of lists, where each sublist contains the cluster assignments for each graph.

    Notes:
    - This function assumes the observation data can be used to create graphs directly correlating to agent positions.
    """
    # Create graphs from the observations
    graphs = create_graphs_from_observation(observation)
    all_clusters = []

    # Apply the Louvain method to each graph
    for G in graphs:
        partition = community_louvain.best_partition(G, weight='weight')
        clusters = [partition[i] for i in range(len(G.nodes()))]
        all_clusters.append(clusters)

    return all_clusters


def create_clusters(observation, n_target, n_agents, agents_per_target):
    """
    Perform hierarchical agglomerative clustering on agent observations to distribute agents 
    across targets, ensuring each target cluster contains a specified number of agents. 
    This function handles agent distributions by adjusting cluster sizes post-clustering 
    to manage imbalances.

    Parameters:
    - observation: initial observations of agents within 
      the multi-agent system environment. Expected to include positional data for each agent.
    - n_target (int): The desired number of target clusters to form.
    - n_agents (int): Total number of agents present in the environment.
    - agents_per_target (int): The specified number of agents to assign to each target cluster.

    Returns:
    - list: A list of lists where each sublist contains adjusted cluster assignments for each 
      graph ensuring each target has the specified number of agents.

    Example:
    >>> env = VmasEnv(scenario='discovery', num_envs=2, n_agents=6, n_targets=2,
                      agents_per_target=3, seed=0, device='cuda')
    >>> observation = env.reset()["agents"]["observation"]  # Reset environment and get initial observations
    >>> clusters = create_clusters(observation, 2, 6, 3)
    >>> print(clusters)
    [[1, 1, 1, 2, 2, 2], [2, 2, 2, 1, 1, 1]]

    Notes:
    - This function makes use of 'scipy.cluster.hierarchy.linkage' and 'fcluster' for clustering.
    - Cluster adjustment involves redistribution of agents among the clusters to correct imbalances, 
      ensuring uniform distribution of agents as specified by 'agents_per_target'.
    """
    if n_agents == 1:
        # Directly return [1] as the cluster assignment for a single agent
        return [[1] for _ in range(len(observation))]
    graphs = create_graphs_from_observation(observation)
    all_clusters = []

    for graph in graphs:
        positions = np.array([graph.nodes[i]['pos'] for i in graph.nodes])
        # Calculate distances using only positions
        distances = calculate_euclidean_distances(positions)

        # Perform hierarchical agglomerative clustering
        Z = linkage(distances, method='ward')
        # Attempt to form exactly n_target clusters
        clusters = fcluster(Z, t=n_target, criterion='maxclust')

        # Post-processing clusters to ensure each target has the specified number of agents
        # This may involve redistributing agents if there are imbalances
        adjusted_clusters = adjust_clusters(clusters, n_target, n_agents, agents_per_target)
        all_clusters.append(adjusted_clusters)

    return all_clusters

def calculate_euclidean_distances(positions):
    from scipy.spatial.distance import pdist, squareform
    # Compute pairwise distances
    return squareform(pdist(positions, 'euclidean'))

def adjust_clusters(clusters, n_target, n_agents, agents_per_target):
    # This function needs to adjust clusters to make sure each has exactly agents_per_target agents
    # This is a placeholder function; actual implementation will depend on specific requirements
    from collections import Counter
    cluster_count = Counter(clusters)
    adjusted_clusters = []

    # Example adjustment: if we have too few agents in some clusters, move some from larger clusters
    for i in range(1, n_target + 1):
        if cluster_count[i] > agents_per_target:
            # Reduce this cluster to agents_per_target
            excess = cluster_count[i] - agents_per_target
            # Reassign excess agents to other clusters that may need more
            # This is a simplification and would need actual indices and more logic
        elif cluster_count[i] < agents_per_target:
            # Increase agents in this cluster from others or note the shortage
            shortage = agents_per_target - cluster_count[i]
            # Actual reassignment logic goes here

    # Return adjusted cluster list
    return clusters  # Placeholder return


from exputil import create_clusters
import torch
from torchrl.envs.transforms import Transform
from tensordict import TensorDict, TensorDictBase
from torchrl.envs.utils import check_env_specs, step_mdp
# Example usage with a TransformedEnv
from torchrl.envs import TransformedEnv, Compose
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.data import BoundedTensorSpec,CompositeSpec, DiscreteTensorSpec
from torchrl.envs.transforms.transforms import _apply_to_composite
class ClusterAssignmentTransform(Transform):
    def __init__(self, n_target, n_agents, agents_per_target, in_key=[("agents","observation")], out_key=[("agents","observation")]):
    #def __init__(self, n_target, n_agents, agents_per_target, in_key=["observation"], out_key=["observation"]):
        super().__init__(in_keys=in_key, out_keys=out_key)
        self.n_target = n_target
        self.n_agents = n_agents
        self.agents_per_target = agents_per_target

    def _apply_transform(self, observation: TensorDictBase) -> None:

        # Call the clustering function
        clusters = create_clusters(observation, self.n_target, self.n_agents, self.agents_per_target)
        clusters_tensor = torch.tensor(clusters[0], dtype=torch.float32, device=observation.device)
        clusters_tensor = clusters_tensor.unsqueeze(0).unsqueeze(-1)  # adding necessary dimensions
        observation = torch.cat([observation, clusters_tensor], dim=-1)
        return observation
    
    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        return self._call(tensordict_reset)





if __name__ == '__main__':
    num_envs = 2
    n_agents = 5
    ted = env.reset()  # Reset environment and get initial observations
    observations = ted['agents']['observation']
    print(observations)

    for env_index in range(num_envs):
        G = nx.Graph()
        positions = observations[env_index, :, :2].cpu().numpy()  # Extract only x, y positions

        # Add nodes with positions
        for i in range(n_agents):
            print(positions[i])
    graphs = create_graphs_from_ted(ted)
    draw_graphs_and_environments(observations, graphs)
    draw_graphs(graphs)


    # Example usage
    # Example environment setup and execution
    # Assuming a function 'env.reset()' returns a TensorDict containing 'agents' -> 'observation' tensor
    n_target = 2  # Number of targets
    agents_per_target = 3  # Number of agents per target
    n_agents=6
    #can't pass below test case, got [array([1, 2], dtype=int32), array([1, 2], dtype=int32)].
    #n_target = 1  # Number of targets
    #agents_per_target = 2  # Number of agents per target
    #n_agents=2
    env = VmasEnv(
        scenario='discovery',
        num_envs=2,
        n_agents=n_agents,
        n_targets=n_target,
        agents_per_target=agents_per_target,
        seed=0,
        device='cuda'
    )
    ted = env.reset()  # Reset environment and get initial observations
    observations = ted['agents']['observation']

    clusters = create_clusters_louvain(observations, n_target,n_agents, agents_per_target)
    print(clusters)

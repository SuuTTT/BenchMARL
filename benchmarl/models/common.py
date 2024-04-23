#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#

import pathlib

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase, TensorDictSequential
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import EnvBase

from benchmarl.utils import _class_from_name, _read_yaml_config, DEVICE_TYPING
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



#from exputil import create_clusters_louvain
import torch
from torchrl.envs.transforms import Transform
from tensordict import TensorDict, TensorDictBase
from torchrl.envs.utils import check_env_specs, step_mdp
# Example usage with a TransformedEnv
from torchrl.envs import TransformedEnv, Compose
from torchrl.envs.libs.vmas import VmasEnv
from torchrl.data import BoundedTensorSpec,CompositeSpec, DiscreteTensorSpec, UnboundedContinuousTensorSpec
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
        clusters = create_clusters_louvain(observation, self.n_target, self.n_agents, self.agents_per_target)
        flat_clusters = [item for sublist in clusters for item in sublist]
        clusters_tensor = torch.tensor(flat_clusters, dtype=torch.float32, device=observation.device)
        clusters_tensor = clusters_tensor.view(observation.shape[0], observation.shape[1], 1)
        observation = torch.cat([observation, clusters_tensor], dim=-1)
        return observation
    def transform_observation_spec(self, observation_spec):
        # Update the observation spec to include the new cluster assignment dimension
        old_spec = observation_spec[("agents", "observation")]
        new_shape = list(old_spec.shape)
        new_shape[-1] += 1  # Adding one for the cluster ID

        new_spec = UnboundedContinuousTensorSpec(
            shape=new_shape,
            dtype=torch.float32,
            device=old_spec.device
        )
        observation_spec[("agents", "observation")] = new_spec
        return observation_spec
    
    def _reset(self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase) -> TensorDictBase:
        return self._call(tensordict_reset)











def _check_spec(tensordict, spec):
    if not spec.is_in(tensordict):
        raise ValueError(f"TensorDict {tensordict} not in spec {spec}")


def parse_model_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    del cfg["name"]
    kwargs = {}
    for key, value in cfg.items():
        if key.endswith("class") and value is not None:
            value = _class_from_name(cfg[key])
        kwargs.update({key: value})
    return kwargs


def output_has_agent_dim(share_params: bool, centralised: bool) -> bool:
    """
    This is a dynamically computed attribute that indicates if the output will have the agent dimension.
    This will be false when share_params==True and centralised==True, and true in all other cases.
    When output_has_agent_dim is true, your model's output should contain the multiagent dimension,
    and the dimension should be absent otherwise

    """
    if share_params and centralised:
        return False
    else:
        return True


class Model(TensorDictModuleBase, ABC):
    """
    Abstract class representing a model.

    Models in BenchMARL are instantiated per agent group.
    This means that each model will process the inputs for a whole group of agents
    They are associated with input and output specs that define their domains.

    Args:
        input_spec (CompositeSpec): the input spec of the model
        output_spec (CompositeSpec): the output spec of the model
        agent_group (str): the name of the agent group the model is for
        n_agents (int): the number of agents this module is for
        device (str): the model's device
        input_has_agent_dim (bool): This tells the model if the input will have a multi-agent dimension or not.
            For example, the input of policies will always have this set to true,
            but critics that use a global state have this set to false as the state is shared by all agents
        centralised (bool): This tells the model if it has full observability.
            This will always be true when ``self.input_has_agent_dim==False``,
            but in cases where the input has the agent dimension, this parameter is
            used to distinguish between a decentralised model (where each agent's data
            is processed separately) and a centralized model, where the model pools all data together
        share_params (bool): This tells the model if it should have only one set of parameters
            or a different set of parameters for each agent.
            This is independent of the other options as it is possible to have different parameters
            for centralized critics with global input.
        action_spec (CompositeSpec): The action spec of the environment
    """

    def __init__(
        self,
        input_spec: CompositeSpec,
        output_spec: CompositeSpec,
        agent_group: str,
        input_has_agent_dim: bool,
        n_agents: int,
        centralised: bool,
        share_params: bool,
        device: DEVICE_TYPING,
        action_spec: CompositeSpec,
    ):
        TensorDictModuleBase.__init__(self)

        self.input_spec = input_spec
        self.output_spec = output_spec
        self.agent_group = agent_group
        self.input_has_agent_dim = input_has_agent_dim
        self.centralised = centralised
        self.share_params = share_params
        self.device = device
        self.n_agents = n_agents
        self.action_spec = action_spec

        self.in_keys = list(self.input_spec.keys(True, True))
        self.out_keys = list(self.output_spec.keys(True, True))

        self.in_key = self.in_keys[0]
        self.out_key = self.out_keys[0]
        self.input_leaf_spec = self.input_spec[self.in_key]
        self.output_leaf_spec = self.output_spec[self.out_key]

        self._perform_checks()

    @property
    def output_has_agent_dim(self) -> bool:
        """
        This is a dynamically computed attribute that indicates if the output will have the agent dimension.
        This will be false when ``share_params==True and centralised==True``, and true in all other cases.
        When output_has_agent_dim is true, your model's output should contain the multi-agent dimension,
        and the dimension should be absent otherwise
        """
        return output_has_agent_dim(self.share_params, self.centralised)

    def _perform_checks(self):
        if not self.input_has_agent_dim and not self.centralised:
            raise ValueError(
                "If input does not have an agent dimension the model should be marked as centralised"
            )

        if len(self.in_keys) > 1:
            raise ValueError("Currently models support just one input key")
        if len(self.out_keys) > 1:
            raise ValueError("Currently models support just one output key")

        if self.agent_group in self.input_spec.keys() and self.input_spec[
            self.agent_group
        ].shape != (self.n_agents,):
            raise ValueError(
                "If the agent group is in the input specs, its shape should be the number of agents"
            )
        if self.agent_group in self.output_spec.keys() and self.output_spec[
            self.agent_group
        ].shape != (self.n_agents,):
            raise ValueError(
                "If the agent group is in the output specs, its shape should be the number of agents"
            )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        # _check_spec(tensordict, self.input_spec)
        tensordict = self._forward(tensordict)
        # _check_spec(tensordict, self.output_spec)
        return tensordict

    ###############################
    # Abstract methods to implement
    ###############################

    @abstractmethod
    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """
        Method to implement for the forward pass of the model.
        It should read self.in_key, process it and write self.out_key.

        Args:
            tensordict (TensorDictBase): the input td

        Returns: the input td with the written self.out_key

        """
        raise NotImplementedError


class SequenceModel(Model):
    """A sequence of :class:`~benchmarl.models.Model`

    Args:
       models (list of Model): the models in the sequence
    """

    def __init__(
        self,
        models: List[Model],
    ):
        super().__init__(
            n_agents=models[0].n_agents,
            input_spec=models[0].input_spec,
            output_spec=models[-1].output_spec,
            centralised=models[0].centralised,
            share_params=models[0].share_params,
            device=models[0].device,
            agent_group=models[0].agent_group,
            input_has_agent_dim=models[0].input_has_agent_dim,
            action_spec=models[0].action_spec,
        )
        self.models = TensorDictSequential(*models)

    def _forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self.models(tensordict)


@dataclass
class ModelConfig(ABC):
    """
    Dataclass representing a :class:`~benchmarl.models.Model` configuration.
    This should be overridden by implemented models.
    Implementors should:

        1. add configuration parameters for their algorithm
        2. implement all abstract methods

    """

    def get_model(
        self,
        input_spec: CompositeSpec,
        output_spec: CompositeSpec,
        agent_group: str,
        input_has_agent_dim: bool,
        n_agents: int,
        centralised: bool,
        share_params: bool,
        device: DEVICE_TYPING,
        action_spec: CompositeSpec,
    ) -> Model:
        """
        Creates the model from the config.

        Args:
            input_spec (CompositeSpec): the input spec of the model
            output_spec (CompositeSpec): the output spec of the model
            agent_group (str): the name of the agent group the model is for
            n_agents (int): the number of agents this module is for
            device (str): the mdoel's device
            input_has_agent_dim (bool): This tells the model if the input will have a multi-agent dimension or not.
                For example, the input of policies will always have this set to true,
                but critics that use a global state have this set to false as the state is shared by all agents
            centralised (bool): This tells the model if it has full observability.
                This will always be true when self.input_has_agent_dim==False,
                but in cases where the input has the agent dimension, this parameter is
                used to distinguish between a decentralised model (where each agent's data
                is processed separately) and a centralized model, where the model pools all data together
            share_params (bool): This tells the model if it should have only one set of parameters
                or a different set of parameters for each agent.
                This is independent of the other options as it is possible to have different parameters
                for centralized critics with global input.
            action_spec (CompositeSpec): The action spec of the environment

        Returns: the Model

        """
        return self.associated_class()(
            **asdict(self),
            input_spec=input_spec,
            output_spec=output_spec,
            agent_group=agent_group,
            input_has_agent_dim=input_has_agent_dim,
            n_agents=n_agents,
            centralised=centralised,
            share_params=share_params,
            device=device,
            action_spec=action_spec,
        )

    @staticmethod
    @abstractmethod
    def associated_class():
        """
        The associated Model class
        """
        raise NotImplementedError

    def process_env_fun(self, env_fun: Callable[[], EnvBase]) -> Callable[[], EnvBase]:
        """
        This function can be used to wrap env_fun
        Args:
            env_fun (callable): a function that takes no args and creates an enviornment

        Returns: a function that takes no args and creates an enviornment

        """
        def wrapped_env() -> EnvBase:
            #from exputil import ClusterAssignmentTransform
            env=env_fun()
            # Setup initial environment (assuming an env object is created already)
            transform = ClusterAssignmentTransform(n_target=0, n_agents=0, agents_per_target=0)
            #env = TransformedEnv(env, Compose([transform]))
            env = TransformedEnv(env, transform)
            return env
        return wrapped_env

    @staticmethod
    def _load_from_yaml(name: str) -> Dict[str, Any]:
        yaml_path = (
            pathlib.Path(__file__).parent.parent
            / "conf"
            / "model"
            / "layers"
            / f"{name.lower()}.yaml"
        )
        cfg = _read_yaml_config(str(yaml_path.resolve()))
        return parse_model_config(cfg)

    @classmethod
    def get_from_yaml(cls, path: Optional[str] = None):
        """
        Load the model configuration from yaml

        Args:
            path (str, optional): The full path of the yaml file to load from.
                If None, it will default to
                benchmarl/conf/model/layers/self.associated_class().__name__

        Returns: the loaded AlgorithmConfig
        """
        if path is None:
            return cls(
                **ModelConfig._load_from_yaml(
                    name=cls.associated_class().__name__,
                )
            )
        else:
            return cls(**parse_model_config(_read_yaml_config(path)))


@dataclass
class SequenceModelConfig(ModelConfig):
    """Dataclass for a :class:`~benchmarl.models.SequenceModel`.


    Examples:

          .. code-block:: python

            import torch_geometric
            from torch import nn
            from benchmarl.algorithms import IppoConfig
            from benchmarl.environments import VmasTask
            from benchmarl.experiment import Experiment, ExperimentConfig
            from benchmarl.models import SequenceModelConfig, GnnConfig, MlpConfig

            experiment = Experiment(
                algorithm_config=IppoConfig.get_from_yaml(),
                model_config=SequenceModelConfig(
                    model_configs=[
                        MlpConfig(num_cells=[8], activation_class=nn.Tanh, layer_class=nn.Linear),
                        GnnConfig(
                            topology="full",
                            self_loops=False,
                            gnn_class=torch_geometric.nn.conv.GraphConv,
                        ),
                        MlpConfig(num_cells=[6], activation_class=nn.Tanh, layer_class=nn.Linear),
                    ],
                    intermediate_sizes=[5, 3],
                ),
                seed=0,
                config=ExperimentConfig.get_from_yaml(),
                task=VmasTask.NAVIGATION.get_from_yaml(),
            )
            experiment.run()

    """

    model_configs: Sequence[ModelConfig]
    intermediate_sizes: Sequence[int]

    def get_model(
        self,
        input_spec: CompositeSpec,
        output_spec: CompositeSpec,
        agent_group: str,
        input_has_agent_dim: bool,
        n_agents: int,
        centralised: bool,
        share_params: bool,
        device: DEVICE_TYPING,
        action_spec: CompositeSpec,
    ) -> Model:
        n_models = len(self.model_configs)
        if not n_models > 0:
            raise ValueError(
                f"SequenceModelConfig expects n_models > 0, got {n_models}"
            )
        if len(self.intermediate_sizes) != n_models - 1:
            raise ValueError(
                f"SequenceModelConfig intermediate_sizes len should be {n_models - 1}, got {len(self.intermediate_sizes)}"
            )

        out_has_agent_dim = output_has_agent_dim(share_params, centralised)
        next_centralised = not out_has_agent_dim
        intermediate_specs = [
            CompositeSpec(
                {
                    f"_{agent_group}_intermediate_{i}": UnboundedContinuousTensorSpec(
                        shape=(n_agents, size) if out_has_agent_dim else (size,)
                    )
                }
            )
            for i, size in enumerate(self.intermediate_sizes)
        ] + [output_spec]

        models = [
            self.model_configs[0].get_model(
                input_spec=input_spec,
                output_spec=intermediate_specs[0],
                agent_group=agent_group,
                input_has_agent_dim=input_has_agent_dim,
                n_agents=n_agents,
                centralised=centralised,
                share_params=share_params,
                device=device,
                action_spec=action_spec,
            )
        ]

        next_models = [
            self.model_configs[i].get_model(
                input_spec=intermediate_specs[i - 1],
                output_spec=intermediate_specs[i],
                agent_group=agent_group,
                input_has_agent_dim=out_has_agent_dim,
                n_agents=n_agents,
                centralised=next_centralised,
                share_params=share_params,
                device=device,
                action_spec=action_spec,
            )
            for i in range(1, n_models)
        ]
        models += next_models
        return SequenceModel(models)

    @staticmethod
    def associated_class():
        return SequenceModel

    def process_env_fun(self, env_fun: Callable[[], EnvBase]) -> Callable[[], EnvBase]:
        for model_config in self.model_configs:
            env_fun = model_config.process_env_fun(env_fun)
        return env_fun

    @classmethod
    def get_from_yaml(cls, path: Optional[str] = None):
        raise NotImplementedError

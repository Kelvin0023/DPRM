import torch
import itertools
import pickle
try:
    # Debug drawing
    from omni.isaac.debug_draw import _debug_draw
    draw = _debug_draw.acquire_debug_draw_interface()
except ImportError:
    pass

class PRM:
    """ Prbabilistic Roadmap (PRM) planner for sampling and planning in the task space """
    def __init__(self, cfg, env, model_target, obs_policy_rms, obs_critic_rms, value_rms, device):
        self.cfg = cfg
        self.env = env
        self.model_target = model_target
        self.obs_policy_rms = obs_policy_rms
        self.obs_critic_rms = obs_critic_rms
        self.value_rms = value_rms
        self.device = device

        # PRM config
        self.prm_samples_per_epoch = cfg["samples_per_epoch"]  # number of samples per epoch
        # assert self.env.num_envs % self.prm_samples_per_epoch == 0
        self.envs_per_sample = self.env.num_envs // self.prm_samples_per_epoch  # environment per sample
        self.new_state_portion = cfg["new_state_portion"]  # portion of new sampled states to be added to the PRM
        self.prm_rollout_len = cfg["rollout_len"]
        self.prm_local_planner = cfg["local_planner"]  # "random" or "policy
        self.visualize_prm = cfg["visualize_prm"]  # visualize the PRM tree in MazeBot and Antmaze tasks
        self.node_merge_threshold = cfg["node_merge_threshold"]  # threshold for merging nodes

        # PRM node and connectivity limit
        self.max_num_nodes = cfg["max_num_nodes"]  # maximum number of nodes in the PRM
        self.max_children_per_node = cfg["max_children"]  # maximum number of children for each node

        # compute the maximum distance in goal that we could extract in random walks
        self.compute_max_goal_dist = cfg["compute_max_goal_dist"]

        """ 
        PRM data structure:
        1. Nodes:
            prm_q: sampled space Q in each node
        2. Edges (uni-directional):
            prm_parents: List of parent for each node, actions are stored from the parent to the node
            prm_children: torch.Tensor of shape (max_num_nodes, max_children_per_node) to store the children of each node
                          initialized with -inf
            children_counter: count the number of children for each node
        """
        # PRM data
        self.prm_q = None  # store the sampled space Q in each node
        self.prm_parents = []  # List of "list of parent" (index) of each node
        self.prm_children = torch.empty(
            (0, self.max_children_per_node)
        )  # Tensor to store the children of each node, default to -inf
        self.children_counter = torch.empty(
            (0, 1),
            dtype=torch.int,
        )  # store the number of children for each node
        self.max_dist_in_goal = torch.empty(
            (0, 1),
            dtype=torch.float,
        )  # store the maximum distance the node could reach within the goal space
        if self.compute_max_goal_dist:
            self.next_node = torch.empty(
                (0, 1),
                dtype=torch.int,
            )  # store the next node to to go which maximize the distance in goal space
        self.prm_obs_policy_buf = torch.empty(
            (0, self.max_children_per_node, self.prm_rollout_len, self.env.cfg.num_observations)
        )  # Tensor to store observation lists along the current node to its children, default to -inf
        self.prm_obs_critic_buf = torch.empty(
            (0, self.max_children_per_node, self.prm_rollout_len, self.env.cfg.num_states)
        )  # Tensor to store state lists along the current node to its children, default to -inf
        self.prm_action_buf = torch.empty(
            (0, self.max_children_per_node, self.prm_rollout_len, self.env.cfg.num_actions)
        )  # Tensor to store action lists along the current node to its children, default to -inf

        # Temporary buffers for storing obs, states and actions between nodes
        self.obs_policy_buf = torch.zeros(
            (self.prm_rollout_len, self.env.num_envs, self.env.cfg.num_observations),
        )
        self.obs_critic_buf = torch.zeros(
            (self.prm_rollout_len, self.env.num_envs, self.env.cfg.num_states),
        )
        self.action_buf = torch.zeros(
            (self.prm_rollout_len, self.env.num_envs, self.env.cfg.num_actions),
        )

        # Temporary variables for building the PRM
        self.x_start_idx = torch.zeros((self.prm_samples_per_epoch,), dtype=torch.int)  # the index of x_start
        self.max_dist = 0.0  # max distance in goal that we could extract in random walks
        self.average_max_dist = 0.0  # average max distance in goal that we could extract in random walks

    def load_prm(self, load_file):
        with open(load_file, "rb") as f:
            graph = pickle.load(f)
        self.prm_q = torch.tensor(graph["prm_q"])
        self.prm_parents = graph["prm_parents"]
        self.prm_children = torch.tensor(graph["prm_children"])
        self.children_counter = torch.tensor(graph["children_counter"])
        self.max_dist_in_goal = torch.tensor(graph["max_dist_in_goal"])
        if self.compute_max_goal_dist:
            self.next_node = torch.tensor(graph["next_node"])
        self.prm_obs_policy_buf = torch.tensor(graph["prm_obs_policy_buf"])
        self.prm_obs_critic_buf = torch.tensor(graph["prm_obs_critic_buf"])
        self.prm_action_buf = torch.tensor(graph["prm_action_buf"])

    def save_prm(self, save_file):
        """ Save the PRM to a file """
        graph = {
            "prm_q": self.prm_q.cpu().numpy(),
            "prm_parents": self.prm_parents,
            "prm_children": self.prm_children.cpu().numpy(),
            "children_counter": self.children_counter.cpu().numpy(),
            "max_dist_in_goal": self.max_dist_in_goal.cpu().numpy(),
            "next_node": self.next_node.cpu().numpy() if self.compute_max_goal_dist else None,
            "prm_obs_policy_buf": self.prm_obs_policy_buf.cpu().numpy(),
            "prm_obs_critic_buf": self.prm_obs_critic_buf.cpu().numpy(),
            "prm_action_buf": self.prm_action_buf.cpu().numpy(),
        }
        with open(save_file, "wb") as f:
            pickle.dump(graph, f)

    def create_child_list(self, num_nodes=1) -> None:
        # Allocate all the required space at once to avoid multiple concatenations
        new_prm_children = torch.full(
            (num_nodes, self.max_children_per_node),
            float('-inf'),
            dtype=torch.float
        )
        new_prm_obs_policy_buf = torch.full(
            (num_nodes, self.max_children_per_node, self.prm_rollout_len, self.env.cfg.num_observations),
            float('-inf'),
            dtype=torch.float
        )
        new_prm_obs_critic_buf = torch.full(
            (num_nodes, self.max_children_per_node, self.prm_rollout_len, self.env.cfg.num_states),
            float('-inf'),
            dtype=torch.float
        )
        new_prm_action_buf = torch.full(
            (num_nodes, self.max_children_per_node, self.prm_rollout_len, self.env.cfg.num_actions),
            float('-inf'),
            dtype=torch.float
        )

        # Concatenate only once for each buffer
        self.prm_children = torch.cat((self.prm_children, new_prm_children), dim=0)
        self.children_counter = torch.cat((self.children_counter, torch.zeros((num_nodes, 1), dtype=torch.int)), dim=0)
        self.max_dist_in_goal = torch.cat((self.max_dist_in_goal, torch.zeros((num_nodes, 1), dtype=torch.float)), dim=0)
        if self.compute_max_goal_dist:
            self.next_node = torch.cat((self.next_node, torch.zeros((num_nodes, 1), dtype=torch.int)), dim=0)
        self.prm_obs_policy_buf = torch.cat((self.prm_obs_policy_buf, new_prm_obs_policy_buf), dim=0)
        self.prm_obs_critic_buf = torch.cat((self.prm_obs_critic_buf, new_prm_obs_critic_buf), dim=0)
        self.prm_action_buf = torch.cat((self.prm_action_buf, new_prm_action_buf), dim=0)

    def run_prm(self) -> None:
        """ Update the PRM with new nodes and edges """
        # Sample new nodes and perform collision check
        self.sample_and_set()
        self.env.simulate()
        self.env.compute_intermediate_values()
        # Rollout for k steps
        self.plan_steps()
        self.add_nodes()

        if self.compute_max_goal_dist:
            self.max_dist = self.max_dist_in_goal.max().item()
            random_idx = torch.randint(0, self.prm_q.size(0), (self.prm_samples_per_epoch,))
            self.average_max_dist = self.max_dist_in_goal[random_idx].mean().item()
            print("*** PRM nodes: ", self.prm_q.shape[0], ", max distance:", self.max_dist, "average max distance", self.average_max_dist, "***")
        else:
            print("*** PRM nodes: ", self.prm_q.shape[0], "***")


    def sample_and_set(self) -> None:
        """ Sample random points in q-space and set the env to these states """
        if self.prm_q is None:
            # Initialize the PRM if it is empty and start the first planning step
            # (All states are sampled randomly in the first step)
            self.start_first_plan()

        else:
            # Start the planning step
            # (Some states are sampled randomly and the rest are sampled from the existing states in PRM)
            self.start_plan()

        assert self.x_start.size(0) == self.prm_samples_per_epoch, "Number of samples should match the PRM samples per epoch"
        assert self.x_start_idx.size(0) == self.prm_samples_per_epoch, "Number of indices should match the PRM samples per epoch"

        # Sample goal states in the task space that are close to the chosen nodes
        self.x_goal = self.env.sample_random_goal_state(num_goal=self.prm_samples_per_epoch).cpu()

        # Set the environment states and goals
        # Initialize empty states and goals
        start_states = torch.zeros((self.env.num_envs, self.env.planning_state_dim), device=self.device)
        if hasattr(self.env, "goal"):
            goals = torch.zeros_like(self.env.goal)


        # Set the environment states to the chosen nodes
        for i in range(self.prm_samples_per_epoch):
            start_states[self.envs_per_sample * i: self.envs_per_sample * (i + 1)] = self.x_start[i]
            # Set the goals to the sampled q_space goals
            if hasattr(self.env, "goal"):
                goals[self.envs_per_sample * i: self.envs_per_sample * (i + 1)] = self.env.q_to_goal(self.x_goal[i].unsqueeze(0))

        # Update the environment states and goals
        with torch.inference_mode():
            self.env.set_env_states(start_states, torch.arange(self.env.num_envs, device=self.device))
            if hasattr(self.env, "goal"):
                self.env.set_goal(goals, torch.arange(self.env.num_envs, device=self.device))
            self.env.simulate()

            # Reset the environment buffer
            self.env.reset_buf[:] = 0
            self.env.reset_terminated[:] = 0
            self.env.reset_time_outs[:] = 0
            self.env.episode_length_buf[:] = 0

    def start_first_plan(self) -> None:
        """ Initialize the empty PRM and start the first planning step """
        # All nodes are sampled randomly in the first step
        self.x_start = self.env.sample_random_nodes(N=self.prm_samples_per_epoch).cpu()

        num_new_nodes = 0
        for i in range(self.prm_samples_per_epoch):
            if self.prm_q is None:  # Initialize the PRM state and q buffers
                self.prm_q = self.x_start[i].unsqueeze(0)
                self.prm_parents.append([])
                num_new_nodes += 1
                self.x_start_idx[i] = 0
            else:
                # compute distance from qbest to other nodes in PRM
                dist_to_x_start = self.env.compute_distance(selected_node=self.x_start[i], prm_nodes=self.prm_q)
                if torch.min(dist_to_x_start) < self.node_merge_threshold:
                    # If the node is too close to the existing nodes, skip
                    closest_idx = torch.argmin(dist_to_x_start)
                    self.x_start_idx[i] = closest_idx
                else:  # Otherwise, add the node to PRM
                    self.prm_q = torch.cat([self.prm_q, self.x_start[i].unsqueeze(0)])
                    self.prm_parents.append([])
                    num_new_nodes += 1
                    self.x_start_idx[i] = self.prm_q.size(0) - 1

        self.create_child_list(num_new_nodes)


    def start_plan(self) -> None:
        """
            Start the planning step by sampling new states and existing states in PRM.
            num_new_sampled_states: self.prm_samples_per_epoch * self.new_state_portion
            num_existing_states: self.prm_samples_per_epoch - num_new_sample
        """

        # Sample new states in the task space
        num_new_sampled_states = int(self.prm_samples_per_epoch * self.new_state_portion)
        self.x_start = self.env.sample_random_nodes(N=num_new_sampled_states).cpu()
        # Sample existing states in PRM
        num_existing_states = self.prm_samples_per_epoch - num_new_sampled_states
        chosen_nodes_idx = torch.randint_like(
            torch.tensor(list(range(num_existing_states))),
            self.prm_q.size(0)
        )
        self.x_start = torch.cat([self.x_start, self.prm_q[chosen_nodes_idx]], dim=0)

        # Compute the index of the new sampled states
        num_new_nodes = 0
        for i in range(num_new_sampled_states):
            # compute distance from qbest to other nodes in PRM
            dist_to_x_start = self.env.compute_distance(selected_node=self.x_start[i], prm_nodes=self.prm_q)
            if torch.min(dist_to_x_start) < self.node_merge_threshold:
                # If the node is too close to the existing nodes, skip
                closest_idx = torch.argmin(dist_to_x_start)
                self.x_start_idx[i] = closest_idx
            else:  # Otherwise, add the node to PRM
                self.prm_q = torch.cat([self.prm_q, self.x_start[i].unsqueeze(0)])
                self.prm_parents.append([])
                num_new_nodes += 1
                self.x_start_idx[i] = self.prm_q.size(0) - 1
        self.create_child_list(num_new_nodes)
        # Add the index of the existing states
        self.x_start_idx[num_new_sampled_states:] = chosen_nodes_idx

    @torch.no_grad()
    def model_act_inference(self, obs_dict):
        # Normalize the observation
        processed_obs = self.obs_policy_rms(obs_dict['policy'])
        # Get predicted actions from target actor
        pred_action_chunks = self.model_target(
            cond={"state": processed_obs},
            deterministic=False,
        )
        return {"actions": pred_action_chunks}

    def plan_steps(self) -> None:
        """ Plan the rollout for the given samples """
        with torch.inference_mode():
            # Fetch the obs and states
            obs_dict = self.env.get_observations()
            # Rollout for k steps
            for k in range(self.prm_rollout_len):
                # Store the initial observation for the critic
                if self.prm_local_planner == "random":
                    # Sample actions randomly
                    pred_next_act = self.env.random_actions()
                elif self.prm_local_planner == "policy":
                    if k % self.prm_rollout_len == 0:
                        res_dict = self.model_act_inference(obs_dict)
                        pred_act_chunk = res_dict["actions"]
                    pred_next_act = pred_act_chunk[:, k % self.prm_rollout_len, :]
                else:
                    raise ValueError("Invalid local planner")
                # Clamp sampled actions and step the environment
                pred_next_act = torch.clamp(pred_next_act, -1.0, 1.0)

                # store intermediate obs and actions
                self.obs_policy_buf[k] = obs_dict["policy"]
                self.obs_critic_buf[k] = obs_dict["critic"]
                self.action_buf[k] = pred_next_act

                # Step the environment
                obs_dict, *_ = self.env.step_without_reset(pred_next_act)

    def add_nodes(self) -> None:
        """Add nodes based on q-sampled"""
        # Collect the reached nodes
        reached_q = self.env.get_env_q().cpu()
        # Evaluate if x_end is valid to be added to the PRM
        invalid, _ = self.env.is_invalid()

        # Reset the new node counter
        self.new_node_num = 0

        # Grow PRM with the reached states (x_end)
        for sample_idx in range(self.prm_samples_per_epoch):
            result = self._add_nodes_for_sample(reached_q, invalid, sample_idx)
            if len(result) > 0:
                state_parent, state_best = result
                self._visualize_new_edges(state_parent, state_best)  # Debug visualization for MazeBot task only

        # Create the new children list for the new nodes
        self.create_child_list(self.new_node_num)

    def _add_nodes_for_sample(
            self,
            reached_q: torch.Tensor,
            invalid: torch.Tensor,
            sample_idx: int
    ) -> tuple:
        """ Add nodes to the PRM tree based on the sampled q-space goal """
        i = sample_idx
        # Fetch the batch of reached states and q
        batch = range(self.envs_per_sample * i, self.envs_per_sample * (i + 1))
        qbatch = reached_q[batch]

        # Extract the obs and actions between nodes in the batch
        obs_policy = self.obs_policy_buf[:, batch, :]
        obs_critic = self.obs_critic_buf[:, batch, :]
        actions = self.action_buf[:, batch, :]

        # Compute distance to the sampled q-space goal
        if hasattr(self.env, "goal"):
            dist = self.env.compute_goal_distance(prm_nodes=qbatch, goal=self.x_goal[i])
        else:
            dist = self.env.compute_distance(selected_node=self.x_goal[i], prm_nodes=qbatch)

        # Find the invalid index in the batch
        invalid_idx = invalid[batch].nonzero(as_tuple=False).squeeze(-1)

        if len(invalid_idx) < len(invalid[batch]):  # there are valid x_end in the batch
            # Find the valid node with the minimum distance to goal
            dist[invalid_idx] = torch.inf  # set the distance to invalid nodes to infinity
            selected_env_id = torch.argmin(dist)
            x_end = qbatch[selected_env_id]

            # compute distance from qbest to other nodes in PRM
            dist_to_x_end = self.env.compute_distance(selected_node=x_end, prm_nodes=self.prm_q)
            parent_idx = self.x_start_idx[i]
            dist_to_x_end[parent_idx] = torch.inf  # ignore the distance to the parent node

            if torch.min(dist_to_x_end) < self.node_merge_threshold:
                # if the node is too close to the existing nodes, we only add the edge
                closest_idx = torch.argmin(dist_to_x_end)
                state_start = self.prm_q[parent_idx].unsqueeze(0)
                state_end = self.prm_q[closest_idx].unsqueeze(0)
                # Add edge (state_start, state_end) to PRM
                self.add_existing_node_edge(
                    parent_idx=parent_idx,
                    existing_node_idx=closest_idx,
                    env_id=selected_env_id,
                    obs_policy_buf=obs_policy,
                    obs_critic_buf=obs_critic,
                    actions_buf=actions
                )

            else:  # Otherwise, we add both the node and the edge
                if self.prm_q.shape[0] > self.max_num_nodes:
                    # Stop adding new nodes if reaching the maximum number of nodes
                    return ()
                else:
                    # Add the new node state_end to PRM
                    self.prm_q = torch.cat([self.prm_q, x_end.unsqueeze(0)])
                    state_start = self.prm_q[parent_idx].unsqueeze(0)
                    state_end = qbatch[selected_env_id].unsqueeze(0)
                    # Add edge (state_start, state_end) to PRM
                    self.add_new_node_edge(
                        parent_idx=parent_idx,
                        env_id=selected_env_id,
                        obs_policy_buf=obs_policy,
                        obs_critic_buf=obs_critic,
                        actions_buf=actions,
                    )
                    self.new_node_num += 1

            return (state_start, state_end)
        else:
            return ()

    def add_existing_node_edge(self, parent_idx, existing_node_idx, env_id, obs_policy_buf, obs_critic_buf, actions_buf) -> None:
        """ Add an edge between the parent and an existing node in the graph """
        # Update the parent list to add the new parent
        self.prm_parents[existing_node_idx].append(parent_idx.item())
        # Update the children list to add the new child
        if self.children_counter[parent_idx] < self.max_children_per_node:
            self.prm_children[parent_idx][self.children_counter[parent_idx]] = float(existing_node_idx)
            # Increment the children counter and update the obs, states and actions buffer
            self.update_counter_and_buffers(parent_idx, env_id, obs_policy_buf, obs_critic_buf, actions_buf)
            # Update the max distance in goal
            if self.compute_max_goal_dist:
                self.update_max_goal_dist(existing_node_idx)

    def add_new_node_edge(self, parent_idx, env_id, obs_policy_buf, obs_critic_buf, actions_buf) -> None:
        """ Add an edge between the parent and a new node in the graph """
        # Update the parent list to add the new parent
        self.prm_parents.append([parent_idx.item()])
        # Update the children list to add the new child
        if self.children_counter[parent_idx] < self.max_children_per_node:
            self.prm_children[parent_idx][self.children_counter[parent_idx]] = self.prm_q.size(0) - 1
            # Increment the children counter and update the obs_policy, obs_critic and actions buffer
            self.update_counter_and_buffers(parent_idx, env_id, obs_policy_buf, obs_critic_buf, actions_buf)
            # Update the max distance in goal
            if self.compute_max_goal_dist:
                self.update_max_goal_dist(self.prm_q.size(0) - 1)

    def update_max_goal_dist(self, new_node_id) -> None:
        # Extract the parent index of the current node
        node_parents = self.prm_parents[new_node_id]
        node_children = [new_node_id for _ in range(len(node_parents))]
        for _ in range(20):
            if len(node_parents) == 0:
                break

            assert len(node_parents) == len(node_children)

            # Compute the distance in goal between the new nodes and these parents
            dist_in_goal = self.env.compute_goal_distance(prm_nodes=self.prm_q[node_parents], goal=self.prm_q[new_node_id])
            # Update the max distance in goal
            self.next_node[node_parents] = torch.where(
                dist_in_goal.unsqueeze(1) > self.max_dist_in_goal[node_parents],
                torch.tensor(node_children, dtype=torch.int).unsqueeze(1),
                self.next_node[node_parents]
            )
            self.max_dist_in_goal[node_parents] = torch.max(self.max_dist_in_goal[node_parents], dist_in_goal.unsqueeze(1))

            # Extract the parent of these parents nodes
            node_grandparents = []
            new_node_children = []

            for i, parent in enumerate(node_parents):
                for grandparent in self.prm_parents[parent]:
                    if grandparent not in node_grandparents:
                        node_grandparents.append(grandparent)
                        new_node_children.append(parent)  # The parent is the child of the grandparent

            # Update the current node parents and node_children for the next iteration
            node_parents = node_grandparents
            node_children = new_node_children

    def update_counter_and_buffers(self, parent_idx, env_id, obs_policy_buf, obs_critic_buf, actions_buf) -> None:
        """ Update the children counter and the obs_policy, obs_critic and actions buffer """
        # Increment the children counter
        self.children_counter[parent_idx] += 1
        # Update the obs and actions buffer
        self.prm_obs_policy_buf[parent_idx][self.children_counter[parent_idx] - 1] = obs_policy_buf[:, env_id, :]
        self.prm_obs_critic_buf[parent_idx][self.children_counter[parent_idx] - 1] = obs_critic_buf[:, env_id, :]
        self.prm_action_buf[parent_idx][self.children_counter[parent_idx] - 1] = actions_buf[:, env_id, :]

    def _visualize_new_edges(
            self,
            state_parent: torch.Tensor,
            state_best: torch.Tensor,
            edge_color: list = [0.7, 0.7, 0.7, 1],
            node_color: list = [1, 1, 0, 1]
    ) -> None:
        """
        Debug visualization: Draw edge for visualization (works only for maze env)
        """
        try:
            if self.visualize_prm:
                p = state_best[0].cpu().numpy()
                parent = state_parent[0].cpu().numpy()

                # draw edge
                point_list_edge_0 = [(parent[0], parent[1], 0.01)] + self.env.scene.env_origins[0].cpu().numpy()
                point_list_edge_1 = [(p[0], p[1], 0.01)] + self.env.scene.env_origins[0].cpu().numpy()
                draw.draw_lines(point_list_edge_0, point_list_edge_1, [edge_color], [1.0])

                # draw "+" marker for the new node
                point_list_node_0 = [(p[0] - 0.01, p[1], 0.01), (p[0], p[1] - 0.01, 0.01)] + self.env.scene.env_origins[0].cpu().numpy()
                point_list_node_1 = [(p[0] + 0.01, p[1], 0.01), (p[0], p[1] + 0.01, 0.01)] + self.env.scene.env_origins[0].cpu().numpy()
                node_colors = [node_color for _ in range(2)]
                node_widths = [1.0 for _ in range(2)]
                draw.draw_lines(point_list_node_0, point_list_node_1, node_colors, node_widths)
        except:
            pass

    def _visualize_nodes(
            self,
            node: torch.Tensor,
            node_color: list = [0, 0, 1, 1]  # blue
    ) -> None:
        """
        Debug visualization: Draw node for visualization (works only for maze env)
        """
        try:
            if self.visualize_prm:
                node_state = node.cpu().numpy()

                # draw "+" marker for the selected node
                point_list_node_0 = [(node_state[0] - 0.02, node_state[1], 0.02), (node_state[0], node_state[1] - 0.02, 0.02)] + self.env.scene.env_origins[0].cpu().numpy()
                point_list_node_1 = [(node_state[0] + 0.02, node_state[1], 0.02), (node_state[0], node_state[1] + 0.02, 0.02)] + self.env.scene.env_origins[0].cpu().numpy()
                node_colors = [node_color for _ in range(2)]
                node_widths = [3.0 for _ in range(2)]
                draw.draw_lines(point_list_node_0, point_list_node_1, node_colors, node_widths)
        except:
            pass

    def perform_search(self, critic, num_searches: int = 2, length: int = 10, search_for_planner=False):
        """ Perform search in the PRM w.r.t the guidance from task critic given a sampled goal """
        search = torch.zeros((num_searches, length, self.prm_q.size(1)), device=self.device)
        obs_policy_buf = torch.zeros(num_searches, length, self.prm_rollout_len, self.env.cfg.num_observations)
        obs_critic_buf = torch.zeros(num_searches, length, self.prm_rollout_len, self.env.cfg.num_states)
        act_buf = torch.zeros(num_searches, length, self.prm_rollout_len, self.env.cfg.num_actions)
        state_buf = torch.zeros(num_searches, length, self.prm_q.size(1))

        if search_for_planner:
            # Sample random goals for the search
            goal = self.env.q_to_planner_goal(self.env.sample_random_goal_state(num_goal=num_searches))
        elif hasattr(self.env, "goal"):
            # Sample random goals for the search
            goal = self.env.q_to_goal(self.env.sample_random_goal_state(num_goal=num_searches))

        # Sample nodes in PRM to start the search
        # Collect the nodes in the PRM with at least one children
        nodes_with_children_index = torch.nonzero(self.children_counter.squeeze() > 0).squeeze()
        self.sampled_idx = nodes_with_children_index[torch.randint(0, nodes_with_children_index.size(0), (num_searches,))]

        # Initialize the mask to keep track of walks that should stop
        zero_children_mask = torch.zeros(num_searches, dtype=torch.bool)

        # Start the search
        for step in range(0, length):
            # Fetch the children list of the selected nodes
            node_children_counter = self.children_counter[self.sampled_idx.int()]

            # Update the mask with OR operation
            zero_children_mask = torch.logical_or(node_children_counter.squeeze() == 0, zero_children_mask)
            # Exit the loop if all search are stopped
            if zero_children_mask.all():
                search[:, step:, :] = float('-inf')
                obs_policy_buf[:, step:, :, :] = float('-inf')
                obs_critic_buf[:, step:, :, :] = float('-inf')
                act_buf[:, step:, :, :] = float('-inf')
                state_buf[:, step:, :] = float('-inf')
                break
            # If the children list is empty, stop the walk
            if zero_children_mask.any():
                search[zero_children_mask, step:, :] = float('-inf')
                obs_policy_buf[zero_children_mask, step:, :, :] = float('-inf')
                obs_critic_buf[zero_children_mask, step:, :, :] = float('-inf')
                act_buf[zero_children_mask, step:, :, :] = float('-inf')
                state_buf[zero_children_mask, step:, :] = float('-inf')

            # Fetch the children list of the nodes with children
            valid_search_idx = self.sampled_idx[~zero_children_mask]
            node_children_list = self.prm_children[valid_search_idx.int()]

            # Select the next node based on the task critic
            if search_for_planner or hasattr(self.env, "goal"):
                valid_search_goal = goal[~zero_children_mask]
                # Select the next node based on the critic
                next_node_idx = self.find_next_node(valid_search_idx.int(), valid_search_goal, critic, select_type="softmax", search_for_planner=search_for_planner)
            else:
                next_node_idx = self.find_next_node(valid_search_idx.int(), None, critic, select_type="softmax", search_for_planner=False)

            # # Fetch index of children with the max goal distance
            # next_node_idx = torch.zeros((valid_search_idx.shape[0], 1), dtype=torch.int64)
            # valid_node_children_counter = node_children_counter[~zero_children_mask]
            # for i in range(valid_search_idx.shape[0]):
            #     node_children = node_children_list[i, :valid_node_children_counter[i].item()].int()
            #     try:
            #         next_node_idx[i] = torch.nonzero(node_children == self.next_node[valid_search_idx[i]], as_tuple=False)[0][0].item()
            #     except:
            #         next_node_idx[i] = 0

            # Fetch index of children with the max goal distance
            self.sampled_idx[~zero_children_mask] = node_children_list.gather(1, next_node_idx).long().squeeze(1)

            # Fill the current step of walk with the sampled nodes
            sampled_state = self.prm_q[self.sampled_idx[~zero_children_mask].int()].to(self.device)
            search[~zero_children_mask, step, :] = sampled_state
            state_buf[~zero_children_mask, step, :] = sampled_state.cpu()

            # Extract the selected obs and actions between the nodes
            sampled_obs_policy = self.prm_obs_policy_buf[valid_search_idx.unsqueeze(1).int(), next_node_idx, :, :].squeeze()
            obs_policy_buf[~zero_children_mask, step, :, :] = sampled_obs_policy
            sampled_obs_critic = self.prm_obs_critic_buf[valid_search_idx.unsqueeze(1).int(), next_node_idx, :, :].squeeze()
            obs_critic_buf[~zero_children_mask, step, :, :] = sampled_obs_critic
            sampled_act = self.prm_action_buf[valid_search_idx.unsqueeze(1).int(), next_node_idx, :, :].squeeze()
            act_buf[~zero_children_mask, step, :, :] = sampled_act

        if search_for_planner or hasattr(self.env, "goal"):
            updated_obs_policy_buf, updated_obs_critic_buf, updated_act_buf, updated_state_buf, updated_goal_buf = self.create_replay_buffer(
                obs_policy_buf,
                obs_critic_buf,
                act_buf,
                goal,
                state_buf,
                search_for_planner=search_for_planner
            )
        else:
            updated_obs_policy_buf, updated_obs_critic_buf, updated_act_buf, updated_state_buf, updated_goal_buf = self.create_replay_buffer(
                obs_policy_buf,
                obs_critic_buf,
                act_buf,
                None,
                state_buf,
                search_for_planner=False
            )
        print("Buffer sizes in the searches: ",
              updated_obs_policy_buf.size(),
              updated_obs_critic_buf.size(),
              updated_act_buf.size(),
              updated_state_buf.size(),
              updated_goal_buf.size() if search_for_planner or hasattr(self.env, "goal") else None
        )

        return search, updated_obs_policy_buf, updated_obs_critic_buf, updated_act_buf, updated_state_buf, updated_goal_buf

    def find_next_node(self, current_node_idx, goal, critic, select_type="softmax", search_for_planner=False) -> torch.Tensor:
        if hasattr(self.env, "goal"):
            critic_goal_start, critic_goal_end = self.env.cfg.goal_idx_critic

        node_children_counter = self.children_counter[current_node_idx]

        # Initialize the probability tensor to select the children
        children_prob_tensor = torch.zeros(node_children_counter.shape[0], self.max_children_per_node)

        if search_for_planner:
            # Extract state and action chunks for the current node
            node_state = self.prm_q[current_node_idx, :].unsqueeze(1)
            node_state = node_state.repeat(1, self.max_children_per_node, 1)
            node_act_chunk = self.prm_action_buf[current_node_idx, :, :, :]
            node_goal = goal.unsqueeze(1).repeat(1, self.max_children_per_node, 1)

            flatten_node_state = node_state.view(-1, self.env.planning_state_dim).to(self.device)
            flatten_node_goal = node_goal.view(-1, self.env.planner_goal_dim).to(self.device)
            flatten_node_state_goal = torch.cat((flatten_node_state, flatten_node_goal), dim=1)
            flatten_node_act_chunk = node_act_chunk.view(-1, self.prm_rollout_len, self.env.cfg.num_actions).to(self.device)
            processed_flatten_node_input = self.state_rms(flatten_node_state_goal)
        else:
            # Extract obs_critic and action chunks for the current node
            obs_critic = self.prm_obs_critic_buf[current_node_idx, :, :, :]
            node_obs_critic = obs_critic[:, :, 0, :]
            node_act_chunk = self.prm_action_buf[current_node_idx, :, :, :]

            # update goal in the observation
            if hasattr(self.env, "goal"):
                node_obs_critic[:, :, critic_goal_start:critic_goal_end] = goal.unsqueeze(1)

            flatten_node_obs_critic = node_obs_critic.view(-1, self.env.cfg.num_states).to(self.device)
            flatten_node_act_chunk = node_act_chunk.view(-1, self.prm_rollout_len, self.env.cfg.num_actions).to(self.device)
            processed_flatten_node_input = self.obs_critic_rms(flatten_node_obs_critic)

        # Predict the Q value of the children
        pred_children_value = critic.sample_min(processed_flatten_node_input, flatten_node_act_chunk)
        pred_children_value = pred_children_value.view(current_node_idx.shape[0], self.max_children_per_node)

        for i in range(current_node_idx.shape[0]):
            valid_children_num = node_children_counter[i]
            pred_valid_children_value = pred_children_value[i, :valid_children_num]
            if select_type == "softmax":
                children_prob_tensor[i, :valid_children_num] = torch.softmax(pred_valid_children_value, dim=0)
            elif select_type == "greedy":
                children_prob_tensor[i, :valid_children_num] = torch.zeros_like(pred_valid_children_value)
                children_prob_tensor[i, torch.argmax(pred_valid_children_value)] = 1.0
            elif select_type == "epsilon_greedy":
                epsilon = 0.1
                children_prob_tensor[i, :valid_children_num] = torch.ones_like(
                    pred_valid_children_value) * epsilon / valid_children_num
                children_prob_tensor[i, torch.argmax(pred_valid_children_value)] = 1.0 - epsilon

        # Create a Categorical distribution with the probabilities
        dist_of_children = torch.distributions.Categorical(children_prob_tensor)
        # Sample one child for each walk
        samples = dist_of_children.sample().unsqueeze(1)

        return samples

    def create_replay_buffer(
            self,
            obs_policy_buf: torch.Tensor,
            obs_critic_buf: torch.Tensor,
            act_buf: torch.Tensor,
            goal_buf: torch.Tensor | None,
            state_buf: torch.Tensor,
            search_for_planner: bool = False
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None):
        # Fetch the indices of the goal
        if search_for_planner:
            goal_size = self.env.planner_goal_dim
        elif hasattr(self.env, "goal"):
            policy_extracted_goal_start, policy_extracted_goal_end = self.env.cfg.extracted_goal_idx_policy
            policy_goal_start, policy_goal_end = self.env.cfg.goal_idx_policy
            critic_goal_start, critic_goal_end = self.env.cfg.goal_idx_critic
            goal_size = policy_goal_end - policy_goal_start

        valid_walk_num = 0
        total_dist_in_goal = 0.0

        # Initialize the empty tensor to hold the observation-action pairs
        updated_obs_policy_buf = torch.empty((0, self.env.cfg.num_observations))
        updated_obs_critic_buf = torch.empty((0, self.env.cfg.num_states))
        updated_act_chunk_buf = torch.empty((0, self.prm_rollout_len, self.env.cfg.num_actions))
        updated_state_buf = torch.empty((0, self.prm_q.size(1)))
        if search_for_planner or hasattr(self.env, "goal"):
            updated_goal_buf = torch.empty((0, goal_size))

        # Find the last valid indices in obs_policy_buf that is not padded with float('-inf')
        last_valid_idx = find_last_valid_indices(obs_policy_buf)
        for search in range(obs_policy_buf.size(0)):
            # Only process the walk if there are valid observations
            if last_valid_idx[search] >= 0:
                valid_obs_policy = obs_policy_buf[search, :last_valid_idx[search] + 1, 0, :]
                valid_obs_critic = obs_critic_buf[search, :last_valid_idx[search] + 1, 0, :]
                valid_act_chunk = act_buf[search, :last_valid_idx[search] + 1, :, :]
                valid_state = state_buf[search, :last_valid_idx[search] + 1, :]
                if search_for_planner:
                    # Add goal to the goal buffer
                    new_goal = goal_buf[search].unsqueeze(0).repeat(valid_obs_policy.size(0), 1).cpu()
                    # new_goal = valid_obs_policy[-1, policy_extracted_goal_start: policy_extracted_goal_end]
                    updated_goal_buf = torch.cat((updated_goal_buf, new_goal), dim=0)
                elif hasattr(self.env, "goal"):
                    # Add goal to the goal buffer
                    new_goal = goal_buf[search].unsqueeze(0).repeat(valid_obs_policy.size(0), 1).cpu()
                    updated_goal_buf = torch.cat((updated_goal_buf, new_goal), dim=0)
                    # Update the goal in the valid observations
                    valid_obs_policy[:, policy_goal_start: policy_goal_end] = new_goal
                    valid_obs_critic[:, critic_goal_start: critic_goal_end] = new_goal
                # Append the valid observations to the updated buffer
                updated_obs_policy_buf = torch.cat((updated_obs_policy_buf, valid_obs_policy), dim=0)
                updated_obs_critic_buf = torch.cat((updated_obs_critic_buf, valid_obs_critic), dim=0)
                # Append the valid q to the updated buffer
                updated_state_buf = torch.cat((updated_state_buf, valid_state), dim=0)
                # Append the valid actions to the updated buffer
                updated_act_chunk_buf = torch.cat((updated_act_chunk_buf, valid_act_chunk), dim=0)

                # Compute the distance in goal between the start and end states
                start_state = valid_state[0].unsqueeze(0)
                end_state = valid_state[-1]
                goal_dist_in_walk = self.env.compute_goal_distance(prm_nodes=start_state, goal=end_state)
                valid_walk_num += 1
                total_dist_in_goal += goal_dist_in_walk

        assert (updated_obs_policy_buf.size(0) == updated_act_chunk_buf.size(0) == updated_obs_critic_buf.size(0) ==
                updated_state_buf.size(0)), \
            "Observations and actions should have the same length"

        print("Average goal distance in the walks: ", total_dist_in_goal / valid_walk_num)

        if search_for_planner or hasattr(self.env, "goal"):
            return (
                updated_obs_policy_buf.to(self.device),
                updated_obs_critic_buf.to(self.device),
                updated_act_chunk_buf.to(self.device),
                updated_state_buf.to(self.device),
                updated_goal_buf.to(self.device)
            )
        else:
            return (
                updated_obs_policy_buf.to(self.device),
                updated_obs_critic_buf.to(self.device),
                updated_act_chunk_buf.to(self.device),
                updated_state_buf.to(self.device),
                None
            )

    def extract_walks(self, num_walks: int = 2, length: int = 10):
        """ Extract random walks from the PRM tree """
        # Initialize the walks tensor
        walks = torch.zeros((num_walks, length, self.prm_q.size(1)), device=self.device)
        obs_policy_buf = torch.zeros((num_walks, length, self.prm_rollout_len, self.env.cfg.num_observations))
        obs_critic_buf = torch.zeros((num_walks, length, self.prm_rollout_len, self.env.cfg.num_states))
        act_buf = torch.zeros((num_walks, length, self.prm_rollout_len, self.env.cfg.num_actions))
        state_buf = torch.zeros((num_walks, length, self.prm_q.size(1)))

        # Sample nodes in PRM to start the walk
        # Collect the nodes in the PRM with at least one children
        nodes_with_children_index = torch.nonzero(self.children_counter.squeeze() > 0).squeeze()
        self.sampled_idx = nodes_with_children_index[torch.randint(0, nodes_with_children_index.size(0), (num_walks,))]

        # Initialize the mask to keep track of walks that should stop
        zero_children_mask = torch.zeros(num_walks, dtype=torch.bool)

        # Traverse the PRM tree to extract the walks
        for step in range(0, length):
            # Fetch the children list of the selected nodes
            selected_children_counter = self.children_counter[self.sampled_idx.int()]

            # Update the mask with the OR operation
            zero_children_mask = torch.logical_or(selected_children_counter.squeeze() == 0, zero_children_mask)
            # Exit the loop if all the nodes have zero children
            if zero_children_mask.all():
                walks[:, step:, :] = float('-inf')
                obs_policy_buf[:, step:, :, :] = float('-inf')
                obs_critic_buf[:, step:, :, :] = float('-inf')
                act_buf[:, step:, :, :] = float('-inf')
                state_buf[:, step:, :] = float('-inf')
                break
            # If the selected node have zero children, stop the walk for that node
            if zero_children_mask.any():
                walks[zero_children_mask, step:, :] = float('-inf')
                obs_policy_buf[zero_children_mask, step:, :, :] = float('-inf')
                obs_critic_buf[zero_children_mask, step:, :, :] = float('-inf')
                act_buf[zero_children_mask, step:, :, :] = float('-inf')
                state_buf[zero_children_mask, step:, :] = float('-inf')

            # Fetch the children list of the nodes with children
            valid_idx = self.sampled_idx[~zero_children_mask]
            selected_children_counter = self.children_counter[valid_idx.int()]
            selected_children_list = self.prm_children[valid_idx.int()]

            # Initiaze the children probabilities tensor
            children_prob_tensor = torch.zeros(selected_children_counter.shape[0], self.max_children_per_node)
            # Create a mask based on the counter values
            mask = (torch.arange(self.max_children_per_node).expand(
                selected_children_counter.shape[0], self.max_children_per_node)
                    < selected_children_counter)
            # Calculate the uniform probabilities for each valid child
            uniform_probabilities = (1.0 / selected_children_counter.float())
            # Expand uniform_probabilities to match the mask's shape
            expanded_probabilities = uniform_probabilities.expand_as(children_prob_tensor)
            # Apply the probabilities to the masked positions
            children_prob_tensor[mask] = expanded_probabilities[mask]

            # Create a Categorical distribution with the probabilities
            dist_of_children = torch.distributions.Categorical(children_prob_tensor)
            # Sample one child for each walk
            samples = dist_of_children.sample().unsqueeze(1)
            self.sampled_idx[~zero_children_mask] = selected_children_list.gather(1, samples).long().squeeze(1)
            # Fill the current step of walk with the sampled nodes
            sampled_state = self.prm_q[self.sampled_idx[~zero_children_mask].int()].to(self.device)
            walks[~zero_children_mask, step, :] = sampled_state
            state_buf[~zero_children_mask, step, :] = sampled_state.cpu()
            # Extract the selected obs and actions between the nodes
            sampled_obs_policy = self.prm_obs_policy_buf[valid_idx.unsqueeze(1).int(), samples, :, :].squeeze()
            obs_policy_buf[~zero_children_mask, step, :, :] = sampled_obs_policy
            sampled_obs_critic = self.prm_obs_critic_buf[valid_idx.unsqueeze(1).int(), samples, :, :].squeeze()
            obs_critic_buf[~zero_children_mask, step, :, :] = sampled_obs_critic
            sampled_act = self.prm_action_buf[valid_idx.unsqueeze(1).int(), samples, :, :].squeeze()
            act_buf[~zero_children_mask, step, :, :] = sampled_act

        (
            updated_obs_policy_buf,
            updated_obs_critic_buf,
            updated_act_buf,
            updated_state_buf,
            goal_buf
        ) = self.create_obs_act_chunk_pairs(obs_policy_buf, obs_critic_buf, act_buf, state_buf)
        print(
            "Buffer sizes: ",
            updated_obs_policy_buf.size(),
            updated_obs_critic_buf.size(),
            updated_act_buf.size(),
            updated_state_buf.size(),
            goal_buf.size() if hasattr(self.env, "goal") else None
        )

        return walks, updated_obs_policy_buf, updated_obs_critic_buf, updated_act_buf, updated_state_buf, goal_buf

    def create_obs_act_chunk_pairs(
            self,
            obs_policy_buf: torch.Tensor,
            obs_critic_buf: torch.Tensor,
            act_buf: torch.Tensor,
            state_buf: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None):
        """ Create observation-action pairs from the Random Walks in PRM tree """
        if hasattr(self.env, "goal"):
            # Fetch the indices of the extracted goal and goal
            policy_extracted_goal_start, policy_extracted_goal_end = self.env.cfg.extracted_goal_idx_policy
            policy_goal_start, policy_goal_end = self.env.cfg.goal_idx_policy
            critic_goal_start, critic_goal_end = self.env.cfg.goal_idx_critic
            goal_size = policy_goal_end - policy_goal_start

        # Initialize the empty tensor to hold the observation-action pairs
        updated_obs_policy_buf = torch.empty((0, self.env.cfg.num_observations))
        updated_obs_critic_buf = torch.empty((0, self.env.cfg.num_states))
        updated_act_chunk_buf = torch.empty((0, self.prm_rollout_len, self.env.cfg.num_actions))
        updated_state_buf = torch.empty((0, self.prm_q.size(1)))
        if hasattr(self.env, "goal"):
            goal_buf = torch.empty((0, goal_size))

        # Find the last valid indices in obs_policy_buf that is not padded with float('-inf')
        last_valid_idx = find_last_valid_indices(obs_policy_buf)
        for walk in range(obs_policy_buf.size(0)):
            # Only process the walk if there are valid observations
            if last_valid_idx[walk] >= 0:
                valid_obs_policy = obs_policy_buf[walk, :last_valid_idx[walk] + 1, :, :]
                valid_obs_critic = obs_critic_buf[walk, :last_valid_idx[walk] + 1, :, :]
                valid_act_chunk = act_buf[walk, :last_valid_idx[walk] + 1, :, :]
                valid_obs_node_policy = valid_obs_policy[:, 0, :]
                valid_obs_node_critic = valid_obs_critic[:, 0, :]
                if hasattr(self.env, "goal"):
                    # Set the goal of the last valid obs to all the valid observations in that walk
                    extracted_goal = valid_obs_policy[-1, -1, policy_extracted_goal_start: policy_extracted_goal_end]
                    # Add goal to the goal buffer
                    all_extracted_goal = extracted_goal.unsqueeze(0).repeat(valid_obs_policy.size(0), 1)
                    goal_buf = torch.cat((goal_buf, all_extracted_goal), dim=0)
                    # Update the goal in the valid observations
                    valid_obs_node_policy[:, policy_goal_start: policy_goal_end] = extracted_goal
                    valid_obs_node_critic[:, critic_goal_start: critic_goal_end] = extracted_goal
                # Append the valid observations to the updated buffer
                updated_obs_policy_buf = torch.cat((updated_obs_policy_buf, valid_obs_node_policy), dim=0)
                updated_obs_critic_buf = torch.cat((updated_obs_critic_buf, valid_obs_node_critic), dim=0)
                # Append the valid q to the updated buffer
                valid_state = state_buf[walk, :last_valid_idx[walk] + 1, :]
                updated_state_buf = torch.cat((updated_state_buf, valid_state), dim=0)
                # Append the valid actions to the updated buffer
                valid_act_chunk = valid_act_chunk.view(-1, self.prm_rollout_len, self.env.cfg.num_actions)
                updated_act_chunk_buf = torch.cat((updated_act_chunk_buf, valid_act_chunk), dim=0)

        assert (updated_obs_policy_buf.size(0) == updated_obs_critic_buf.size(0) == updated_act_chunk_buf.size(0) ==
                updated_state_buf.size(0)), \
            "Observations and actions should have the same length"
        if hasattr(self.env, "goal"):
            return (
                updated_obs_policy_buf.to(self.device),
                updated_obs_critic_buf.to(self.device),
                updated_act_chunk_buf.to(self.device),
                updated_state_buf.to(self.device),
                goal_buf.to(self.device)
            )
        else:
            return (
                updated_obs_policy_buf.to(self.device),
                updated_obs_critic_buf.to(self.device),
                updated_act_chunk_buf.to(self.device),
                updated_state_buf.to(self.device),
                None
            )

    def extract_data(self, buffer_size: int = 1024):
        # Collect the nodes in the PRM with at least one children
        nodes_with_children_idx = torch.nonzero(self.children_counter.squeeze() > 0).squeeze()
        # Sample random nodes with at least one children in PRM
        sampled_nodes_idx = nodes_with_children_idx[torch.randint(0, nodes_with_children_idx.size(0), (buffer_size,))]

        # Randomly sample a child from the children list of the sampled nodes
        sampled_nodes_children_counter = self.children_counter[sampled_nodes_idx].squeeze()
        sampled_nodes_children = torch.floor(
            torch.rand_like(sampled_nodes_children_counter, dtype=torch.float) * sampled_nodes_children_counter).to(
            torch.int)

        # Extract the obs and action chunks
        sampled_obs_policy = self.prm_obs_policy_buf[sampled_nodes_idx, sampled_nodes_children, 0, :]
        sampled_obs_critic = self.prm_obs_critic_buf[sampled_nodes_idx, sampled_nodes_children, 0, :]
        sampled_action_chunk = self.prm_action_buf[sampled_nodes_idx, sampled_nodes_children, :, :]
        # Extract x_sample
        sampled_state = self.prm_q[sampled_nodes_idx]
        if hasattr(self.env, "goal"):
            # Sample random goals for each data point
            sampled_goal = self.env.q_to_goal(self.env.sample_random_goal_state(num_goal=buffer_size))

        if hasattr(self.env, "goal"):
            return sampled_obs_policy.to(self.device), sampled_obs_critic.to(self.device), sampled_action_chunk.to(
                self.device), sampled_state.to(self.device), sampled_goal.to(self.device)
        else:
            return sampled_obs_policy.to(self.device), sampled_obs_critic.to(self.device), sampled_action_chunk.to(
                self.device), sampled_state.to(self.device), None

    def extract_demos(self, num_demos: int = 2, max_len: int = 10, num_parents: int = 5):
        obs_policy_buf = torch.empty((0, self.env.cfg.num_observations))
        obs_critic_buf = torch.empty((0, self.env.cfg.num_states))
        act_buf = torch.empty((0, self.prm_rollout_len, self.env.cfg.num_actions))

        for i in range(num_demos):
            results = self.extract_one_demo(max_len, num_parents)

            if results is not None:
                extracted_obs_policy, extracted_obs_critic, extracted_act = results
                obs_policy_buf = torch.cat((obs_policy_buf, extracted_obs_policy), dim=0)
                obs_critic_buf = torch.cat((obs_critic_buf, extracted_obs_critic), dim=0)
                act_buf = torch.cat((act_buf, extracted_act), dim=0)

        return obs_policy_buf, obs_critic_buf, act_buf

    def extract_one_demo(self, max_len, num_parents):
        obs_policy_buf = torch.empty((0, self.env.cfg.num_observations))
        obs_critic_buf = torch.empty((0, self.env.cfg.num_states))
        act_buf = torch.empty((0, self.prm_rollout_len, self.env.cfg.num_actions))

        # Sample a random node in PRM that serves as goal
        goal_node_idx = torch.randint(0, self.prm_q.size(0), (1,)).item()
        node_parent_idx_list = self.prm_parents[goal_node_idx]

        # visualize the goal node
        if self.visualize_prm:
            self._visualize_nodes(node=self.prm_q[goal_node_idx], node_color=[1, 0, 0, 1])

        # Extract the goal and extracted goal indices
        new_goal = self.env.q_to_goal(self.prm_q[goal_node_idx].unsqueeze(0))
        policy_goal_start, policy_goal_end = self.env.cfg.goal_idx_policy
        critic_goal_start, critic_goal_end = self.env.cfg.goal_idx_critic

        # Terminate if the node has no parent
        if len(node_parent_idx_list) == 0:
            return None

        # Initialize the lists to keep track of the parents and children
        nodes_list = []
        dist_list = []

        for step in range(max_len):
            if step == 0:
                # Extract top k parents with the largest goal_dist
                dist_thres = 0
                results = self.select_best_parents(
                    goal_node_idx,
                    node_parent_idx_list,
                    dist_thres,
                    num_parents
                )
                if results is None:
                    break
                else:
                    selected_parents_idx, dist_parents = results

                    # visualize the selected node
                    if self.visualize_prm:
                        for i in range(len(selected_parents_idx)):
                            self._visualize_nodes(node=self.prm_q[selected_parents_idx[i]], node_color=[0, 0, 1, 1])

                    # Extract the stored data
                    (
                        obs_policy,
                        obs_critic,
                        actions
                    ) = self.find_stored_data(goal_node_idx, selected_parents_idx)
                    # Update the buffer
                    obs_policy_buf = torch.cat((obs_policy_buf, obs_policy), dim=0)
                    obs_critic_buf = torch.cat((obs_critic_buf, obs_critic), dim=0)
                    act_buf = torch.cat((act_buf, actions), dim=0)
                    # Update the lists
                    nodes_list = selected_parents_idx
                    dist_list = dist_parents.tolist()
                    assert len(nodes_list) == len(dist_list)

            else:
                # Create buffers for the next time step
                next_nodes_list = []
                next_dist_list = []

                for i in range(len(nodes_list)):
                    node_child_idx = nodes_list[i]
                    node_parent_idx_list = self.prm_parents[node_child_idx]

                    dist_thres = dist_list[i]
                    results = self.select_best_parents(
                        goal_node_idx,
                        node_parent_idx_list,
                        dist_thres,
                        num_parents
                    )
                    if results is None:
                        break
                    else:
                        selected_parents_idx, dist_parents = results

                        # visualize the selected node
                        if self.visualize_prm:
                            for i in range(len(selected_parents_idx)):
                                self._visualize_nodes(node=self.prm_q[selected_parents_idx[i]], node_color=[0, 0, 1, 1])

                        # Extract the stored data
                        (
                            obs_policy,
                            obs_critic,
                            actions
                        ) = self.find_stored_data(node_child_idx, selected_parents_idx)
                        # Update the buffer
                        obs_policy_buf = torch.cat((obs_policy_buf, obs_policy), dim=0)
                        obs_critic_buf = torch.cat((obs_critic_buf, obs_critic), dim=0)
                        act_buf = torch.cat((act_buf, actions), dim=0)
                        # Update the lists
                        next_nodes_list += selected_parents_idx
                        next_dist_list += dist_parents.tolist()
                        assert len(next_nodes_list) == len(next_dist_list)

                # Reset the lists
                nodes_list = next_nodes_list
                dist_list = next_dist_list


        # Update the obs buffers with the new goal
        obs_policy_buf[:, policy_goal_start:policy_goal_end] = new_goal
        obs_critic_buf[:, critic_goal_start:critic_goal_end] = new_goal

        return obs_policy_buf, obs_critic_buf, act_buf

    # def select_best_parents(self, node_idx, parent_idx_list, dist_thres, num_parents):
    #     # Compute goal_dist between the node and its parents
    #     goal_dist = self.env.compute_goal_distance(prm_nodes=self.prm_q[parent_idx_list], goal=self.prm_q[node_idx])
    #     # Return None if the dist_thres is not satisfied
    #     if goal_dist.max() < dist_thres:
    #         return None
    #     # Reduce the number of parents to extract if we do not have enough parents
    #     if len(goal_dist) < num_parents:
    #         num_parents = len(goal_dist)
    #     # Select (num_parents) nodes with the highest goal_dist
    #     top_parents_dist, selected_parents_list_idx = torch.topk(goal_dist, num_parents)
    #     # Remove the nodes if the dist_thres is not satisfied
    #     selected_parents_list_idx = selected_parents_list_idx[goal_dist[selected_parents_list_idx] > dist_thres]
    #     print("Debug: top_parents_dist: ", top_parents_dist)
    #     print("Debug: selected_parents_list_idx: ", selected_parents_list_idx)
    #     print("Debug: goal_dist[selected_parents_list_idx] > dist_thres:", goal_dist[selected_parents_list_idx] > dist_thres)
    #     print("Debug: goal_dist[selected_parents_list_idx]: ", goal_dist[selected_parents_list_idx])
    #     top_parents_dist = top_parents_dist[goal_dist[selected_parents_list_idx] >= dist_thres]
    #     # Extract the index of parent nodes in the graph
    #     parents_idx = [parent_idx_list[i] for i in selected_parents_list_idx.tolist()]
    #     return parents_idx, top_parents_dist

    def select_best_parents(self, node_idx, parent_idx_list, dist_thres, num_parents):
        # Compute goal_dist between the node and its parents
        goal_dist = self.env.compute_goal_distance(prm_nodes=self.prm_q[parent_idx_list], goal=self.prm_q[node_idx])

        # Return None if the dist_thres is not satisfied or there are no parents for the current node
        if goal_dist.numel() == 0 or goal_dist.max() < dist_thres:
            return None

        # Reduce the number of parents to extract if we do not have enough parents
        if len(goal_dist) < num_parents:
            num_parents = len(goal_dist)

        # Select (num_parents) nodes with the highest goal_dist
        top_parents_dist, selected_parents_list_idx = torch.topk(goal_dist, num_parents)

        # Filter indices where goal_dist is greater than dist_thres
        valid_mask = goal_dist[selected_parents_list_idx] > dist_thres
        selected_parents_list_idx = selected_parents_list_idx[valid_mask]
        top_parents_dist = top_parents_dist[valid_mask]

        # Extract the index of parent nodes in the graph
        parents_idx = [parent_idx_list[i] for i in selected_parents_list_idx.tolist()]
        return parents_idx, top_parents_dist

    def find_stored_data(self, child_node_idx, parent_node_idx_list):
        obs_policy_buf = torch.empty((0, self.env.cfg.num_observations))
        obs_critic_buf = torch.empty((0, self.env.cfg.num_states))
        act_buf = torch.empty((0, self.prm_rollout_len, self.env.cfg.num_actions))

        # Find the index of node_idx in each child list of the parents
        for i in range(len(parent_node_idx_list)):
            # find the index of node_idx in the child list of the parent
            child_list = self.prm_children[parent_node_idx_list[i]]
            # print("Debug: child_list: ", child_list)
            # print("Debug: node_idx: ", child_node_idx)
            child_idx = torch.nonzero(child_list == child_node_idx, as_tuple=False)[0].item()
            # Extract the obs and action chunks
            obs_policy_buf = torch.cat(
                (obs_policy_buf, self.prm_obs_policy_buf[parent_node_idx_list[i], child_idx, 0, :].unsqueeze(0)),
                dim=0
            )
            obs_critic_buf = torch.cat(
                (obs_critic_buf, self.prm_obs_critic_buf[parent_node_idx_list[i], child_idx, 0, :].unsqueeze(0)),
                dim=0
            )
            act_buf = torch.cat(
                (act_buf, self.prm_action_buf[parent_node_idx_list[i], child_idx, :, :].unsqueeze(0)),
                dim=0
            )
        return obs_policy_buf, obs_critic_buf, act_buf


def find_last_valid_indices(obs_policy_buf):
    """ Find the last valid indices in obs_policy_buf not padded with float('-inf') """
    # Create a mask where valid entries are True
    valid_mask = (obs_policy_buf != float('-inf'))

    # Reduce the mask along the dimensions that we are not interested in
    # For obs_policy_buf with shape (num_walks, length - 1, prm_rollout_len, num_observations),
    # we need to reduce it to (num_walks, length - 1) to find the last valid index in the second dimension.
    reduced_valid_mask = valid_mask.any(dim=(2, 3))

    # Initialize a tensor to hold the last valid indices, filled with -1 initially
    last_valid_indices = torch.full((obs_policy_buf.size(0), obs_policy_buf.size(1)), -1, device=obs_policy_buf.device)

    # Find the indices where valid entries occur
    valid_indices = torch.arange(obs_policy_buf.size(1), device=obs_policy_buf.device).unsqueeze(0).expand_as(reduced_valid_mask)

    # Mask out the valid indices
    last_valid_indices[reduced_valid_mask] = valid_indices[reduced_valid_mask]

    # For each walk, find the max index where valid entry occurs
    last_valid_indices = last_valid_indices.max(dim=1)[0]

    return last_valid_indices
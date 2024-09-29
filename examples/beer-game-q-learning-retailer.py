import numpy as np
import gym
from gym import spaces

class BeerGameEnv(gym.Env):
    def __init__(self):
        super(BeerGameEnv, self).__init__()

        # Number of agents: Retailer, Wholesaler, Distributor, Manufacturer
        self.num_agents = 4

        # Action space: how much to order from the upstream agent (0-100 units)
        self.action_space = spaces.Discrete(101)

        # Observation space for the retailer: inventory, pending orders, incoming shipment
        self.observation_space = spaces.Box(low=0, high=1000, shape=(3,), dtype=np.int32)

        # Supply chain parameters
        self.max_inventory = 1000  # max inventory for each agent
        self.max_order = 100       # max order per timestep
        self.lead_time = 2         # delay in shipments
        self.holding_cost = 1      # cost per unit of inventory held
        self.stockout_cost = 2     # cost per unit of unmet demand

        # Add a maximum number of steps per episode
        self.max_steps = 200  # You can adjust this based on the complexity you want
        self.current_step = 0  # To track the current step

        # Initialize state
        self.reset()

    def reset(self):
        # Reset the environment to an initial state

        # Initial inventory for each agent (randomized)
        self.inventory = np.random.randint(50, 100, size=self.num_agents)

        # Pending orders (empty initially)
        self.pending_orders = np.zeros(self.num_agents, dtype=np.int32)

        # Incoming shipments (delayed by lead_time)
        self.incoming_shipments = np.zeros((self.num_agents, self.lead_time), dtype=np.int32)

        # Retailer demand (randomized)
        self.retailer_demand = np.random.randint(20, 50)

        # Reset step counter
        self.current_step = 0

        return self._get_observation()

    def _get_observation(self):
        # Return the state for the retailer: inventory, pending orders, incoming shipments
        return np.array([self.inventory[0], self.pending_orders[0], self.incoming_shipments[0, 0]])

    def step(self, action):
        rewards = np.zeros(self.num_agents)

        # Step 1: Process incoming shipments for each agent
        self._process_incoming_shipments()

        # Step 2: Process customer demand at the retailer level
        rewards[0] += self._process_customer_demand()

        # Step 3: Update inventories and rewards for each agent
        for agent in range(1, self.num_agents):
            self._process_agent(agent, action if agent == 0 else self.pending_orders[agent])

        # Step 4: Handle orders from each agent and update pending orders
        self._place_orders(action)

        # Calculate rewards (negative cost) for the retailer
        rewards[0] -= self.holding_cost * self.inventory[0]  # holding cost for the retailer

        # Increment the step counter
        self.current_step += 1

        # Check if episode is done (terminate if max_steps is reached)
        done = self.current_step >= self.max_steps

        return self._get_observation(), rewards[0], done, {}

    def _process_incoming_shipments(self):
        # Move incoming shipments one step closer (lead_time reduction)
        for agent in range(self.num_agents):
            self.inventory[agent] += self.incoming_shipments[agent, 0]  # Add incoming shipment to inventory
            self.incoming_shipments[agent, :-1] = self.incoming_shipments[agent, 1:]  # Shift the shipment pipeline
            self.incoming_shipments[agent, -1] = 0  # Reset the last spot in shipment pipeline

    def _process_customer_demand(self):
        # Retailer fulfills customer demand
        fulfilled_demand = min(self.inventory[0], self.retailer_demand)
        unmet_demand = self.retailer_demand - fulfilled_demand
        self.inventory[0] -= fulfilled_demand

        # Retailer incurs stockout costs if demand is unmet
        stockout_cost = unmet_demand * self.stockout_cost
        reward = -stockout_cost

        # Randomly update the next retailer demand for the next step
        self.retailer_demand = np.random.randint(20, 50)

        return reward

    def _process_agent(self, agent, action):
        # Agents (Wholesaler, Distributor, Manufacturer) process orders from downstream
        demand_from_downstream = self.pending_orders[agent - 1]
        fulfilled_demand = min(self.inventory[agent], demand_from_downstream)
        unmet_demand = demand_from_downstream - fulfilled_demand

        # Update inventory
        self.inventory[agent] -= fulfilled_demand

        # Update pending orders (delayed)
        self.pending_orders[agent] = action

    def _place_orders(self, action):
        # Retailer places an order to the wholesaler
        self.incoming_shipments[1, -1] = action  # Set orders as incoming shipment (delayed)

    def render(self, mode='human'):
        print(f"Inventory: {self.inventory}")
        print(f"Pending Orders: {self.pending_orders}")
        print(f"Incoming Shipments: {self.incoming_shipments[:, 0]}")



# Q-Learning Algorithm
class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.1):
        self.env = env
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Initialize Q-table with zeros
        # Add an extra dimension for the actions (101 possible actions)
        self.q_table = np.zeros((1001, 101, 101, env.action_space.n))  # (state1, state2, state3, action)

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            # Exploration: random action
            return self.env.action_space.sample()
        else:
            # Exploitation: choose the best action from Q-table
            return np.argmax(self.q_table[state[0], state[1], state[2]])

    def update_q_table(self, state, action, reward, next_state):
        current_q = self.q_table[state[0], state[1], state[2], action]
        best_next_q = np.max(self.q_table[next_state[0], next_state[1], next_state[2]])
        # Q-learning update rule
        self.q_table[state[0], state[1], state[2], action] = current_q + self.alpha * (reward + self.gamma * best_next_q - current_q)

    def decay_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay


# Training the agent
if __name__ == "__main__":
    env = BeerGameEnv()
    agent = QLearningAgent(env)

    episodes = 1000
    for episode in range(episodes):
        state = env.reset()

        total_reward = 0
        done = False
        while not done:
            # Choose action based on current state
            action = agent.choose_action(state)

            # Take action in the environment
            next_state, reward, done, _ = env.step(action)

            # Update Q-table
            agent.update_q_table(state, action, reward, next_state)

            # Move to the next state
            state = next_state

            # Accumulate reward
            total_reward += reward

        # Decay exploration rate
        agent.decay_epsilon()

        if episode % 100 == 0:
            print(f"Episode {episode}: Total Reward: {total_reward}, Epsilon: {agent.epsilon}")

    print("Training complete.")

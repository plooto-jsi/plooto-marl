import numpy as np
import gym
from gym import spaces

class BeerGameEnv(gym.Env):
    def __init__(self):
        super(BeerGameEnv, self).__init__()

        # Number of agents: Retailer, Wholesaler, Distributor, Manufacturer
        self.num_agents = 4

        # Action space: how much to order from the upstream agent (0-100 units)
        self.action_space = spaces.MultiDiscrete([101] * self.num_agents)

        # Observation space for each agent: current inventory, pending orders, incoming shipment
        self.observation_space = spaces.Box(low=0, high=1000, shape=(self.num_agents, 3), dtype=np.int32)

        # Supply chain parameters
        self.max_inventory = 1000  # max inventory for each agent
        self.max_order = 100       # max order per timestep
        self.lead_time = 2         # delay in shipments
        self.holding_cost = 1      # cost per unit of inventory held
        self.stockout_cost = 2     # cost per unit of unmet demand

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

        return self._get_observation()

    def _get_observation(self):
        # Return the state for each agent: inventory, pending orders, incoming shipments
        return np.column_stack((self.inventory, self.pending_orders, self.incoming_shipments[:, 0]))

    def step(self, actions):
        # Actions are the orders each agent places to its upstream agent
        assert len(actions) == self.num_agents

        rewards = np.zeros(self.num_agents)

        # Step 1: Process incoming shipments for each agent
        self._process_incoming_shipments()

        # Step 2: Process customer demand at the retailer level
        rewards[0] += self._process_customer_demand()

        # Step 3: Update inventories and rewards for each agent
        for agent in range(1, self.num_agents):
            self._process_agent(agent, actions[agent])

        # Step 4: Handle orders from each agent and update pending orders
        self._place_orders(actions)

        # Calculate rewards (negative cost)
        for agent in range(self.num_agents):
            rewards[agent] -= self.holding_cost * self.inventory[agent]  # holding cost

        done = False  # This game can run indefinitely; you can choose when to stop
        return self._get_observation(), rewards, done, {}

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

    def _place_orders(self, actions):
        # Each agent places orders to the upstream agent
        for agent in range(self.num_agents - 1):
            order_quantity = actions[agent + 1]
            self.incoming_shipments[agent + 1, -1] = order_quantity  # Set orders as incoming shipment (delayed)

    def render(self, mode='human'):
        print(f"Inventory: {self.inventory}")
        print(f"Pending Orders: {self.pending_orders}")
        print(f"Incoming Shipments: {self.incoming_shipments[:, 0]}")

# Example usage
if __name__ == "__main__":
    env = BeerGameEnv()
    obs = env.reset()

    for _ in range(5):  # Simulate 5 steps
        actions = env.action_space.sample()  # Random actions
        obs, rewards, done, info = env.step(actions)
        env.render()
        print(f"Rewards: {rewards}")
        print('-' * 30)

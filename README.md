# plooto-marl
Multi-agent reinforcement learning for Plooto.

## Beer Game Example

Location: `example/beer-game.py`

### Key Features:
1. Agents and Inventory: The environment models the four main agents (Retailer, Wholesaler, Distributor, and Manufacturer). Each agent manages inventory and places orders upstream.
2. Action Space: Each agent's action is the number of units they order from their upstream supplier (capped at 100 for simplicity, but this can be adjusted).
3. Shipment and Lead Time: Each order takes 2 time steps to arrive (configurable via lead_time), simulating real-world shipment delays.
4. State Representation: Each agent observes its current inventory, pending orders, and incoming shipments, encapsulating the key decision-making factors in a supply chain.
5. Reward Mechanism: The environment includes penalties for inventory holding costs and stockout costs (when an agent cannot fulfill orders due to insufficient inventory). The goal is to minimize the total cost incurred by the agents.

### Steps to Extend:
1. Randomized Lead Times: You can randomize the lead times to simulate transportation variability.
2. Stochastic Demand: Demand can be further varied to reflect real-world fluctuations.
3. Customizable Parameters: You can scale this to multiple product types or different transportation modes (e.g., fast but expensive vs slow but cheap).
4. Multiple Episodes: Integrate this into a reinforcement learning training loop to allow agents to learn optimal policies over multiple episodes.
5. This environment provides a solid foundation to test MARL algorithms such as Independent Q-learning, QMIX, or MADDPG, depending on the complexity you want to address. Would you like guidance on the RL algorithm to use with this environment?

### Beergame Links

* https://github.com/transentis/bptk_py_tutorial/blob/master/model_library/beergame/training_ai_beergame.ipynb
* https://www.transentis.com/blog/training-artificial-intelligence-to-play-the-beer-game

## Reinforcement Learning Links

* https://docs.cleanrl.dev/

## Supply Chain RL

* https://www.dfki.de/fileadmin/user_upload/import/14930_ifacconf_preprint.pdf
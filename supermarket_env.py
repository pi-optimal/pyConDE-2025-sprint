import gymnasium as gym
import numpy as np
from gymnasium import spaces
import pandas as pd


class SupermarketEnv(gym.Env):
    """
    Supermarket Inventory Management Environment
    
    This environment simulates the inventory management of perishable products in a supermarket.
    The goal is to optimize ordering quantities to minimize waste while ensuring product availability.
    
    State:
        - Current inventory levels (for each day of shelf life remaining)
        - Day of week (0-6)
        - Recent demand history
        
    Action:
        - Order quantity for new inventory
        
    Reward:
        - Profit = Revenue - Costs
        - Revenue from sales
        - Costs include:
          * Purchase cost
          * Holding cost
          * Stockout cost (lost sales)
          * Waste cost (expired products)
    """
    
    metadata = {'render.modes': ['human']}

    def __init__(self, config=None):
        """
        Initialize the supermarket environment with configuration parameters.
        
        Args:
            config (dict): Configuration parameters for the environment
        """
        # Default configuration
        self.default_config = {
            'max_inventory': 100,         # Maximum inventory capacity
            'shelf_life': 5,              # Shelf life of products in days
            'purchase_cost': 1.0,         # Cost to purchase one unit
            'selling_price': 2.5,         # Price at which one unit is sold
            'holding_cost': 0.1,          # Daily cost to hold one unit
            'stockout_cost': 1.0,         # Cost per stockout
            'waste_cost': 0.5,            # Additional cost per wasted unit
            'demand_mean': 15,            # Mean of daily demand
            'demand_std': 5,              # Standard deviation of daily demand
            'weekday_factors': [0.8, 0.7, 0.9, 1.0, 1.2, 1.5, 1.3],  # Demand modifier by weekday
            'episode_length': 30,         # Length of an episode in days
            'seed': None                  # Random seed
        }
        
        # Update with provided configuration
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        # Set random seed
        self.np_random = np.random.RandomState(self.config['seed'])
        
        # Space definitions
        # State space: [inventory_level (per day of shelf life) + day_of_week + demand_history]
        inventory_space = spaces.Box(
            low=0, 
            high=self.config['max_inventory'], 
            shape=(self.config['shelf_life'],), 
            dtype=np.int32
        )
        
        day_space = spaces.Discrete(7)  # Day of week (0-6)
        
        # We'll keep track of the last 7 days of demand
        demand_history_space = spaces.Box(
            low=0, 
            high=100, 
            shape=(7,), 
            dtype=np.int32
        )
        
        # Combine all state components
        self.observation_space = spaces.Dict({
            'inventory': inventory_space,
            'day_of_week': spaces.Box(low=0, high=6, shape=(1,), dtype=np.int32),
            'demand_history': demand_history_space
        })
        
        # Action space: order quantity
        self.action_space = spaces.Box(
            low=0, 
            high=self.config['max_inventory'], 
            shape=(1,), 
            dtype=np.int32
        )
        
        # Initialize state variables
        self.reset()
        
    def reset(self, seed=None, options=None):
        """Reset the environment to an initial state."""
        # Set seed if provided
        if seed is not None:
            self.np_random = np.random.RandomState(seed)
            
        # Reset episode counter
        self.current_day = 0
        self.day_of_week = 0  # Start on Monday (0)
        
        # Initialize inventory (start with random inventory levels)
        self.inventory = np.zeros(self.config['shelf_life'], dtype=np.int32)
        initial_stock = self.np_random.randint(5, 20)
        # Distribute initial stock among different shelf-life days
        distribution = self.np_random.dirichlet(np.ones(self.config['shelf_life'])) 
        self.inventory = np.round(distribution * initial_stock).astype(np.int32)
        
        # Initialize demand history with some random values based on demand distribution
        self.demand_history = np.zeros(7, dtype=np.int32)
        for i in range(7):
            day_factor = self.config['weekday_factors'][i]
            mean_demand = self.config['demand_mean'] * day_factor
            self.demand_history[i] = max(0, int(self.np_random.normal(mean_demand, self.config['demand_std'])))
        
        # Sales, waste and stockout tracking
        self.total_sales = 0
        self.total_waste = 0
        self.total_stockouts = 0
        
        # Additional metrics for the episode
        self.episode_history = {
            'day': [],
            'day_of_week': [],
            'demand': [],
            'order_quantity': [],
            'inventory_before_order': [],
            'inventory_after_order': [],
            'inventory_after_demand': [],
            'sales': [],
            'waste': [],
            'stockouts': [],
            'revenue': [],
            'purchase_cost': [],
            'holding_cost': [],
            'stockout_cost': [],
            'waste_cost': [],
            'profit': [],
            'total_inventory': []
        }
        
        # Create the initial observation
        observation = self._get_observation()
        
        # Additional info
        info = {}
        
        return observation, info
    
    def step(self, action):
        """
        Take a step in the environment by ordering inventory and processing a day's operations.
        
        Args:
            action: The order quantity (can be float, will be rounded to int)
            
        Returns:
            observation: The current state
            reward: The reward from this step
            terminated: Whether the episode is over
            truncated: Whether the episode was truncated
            info: Additional information
        """
        # Process the action (order quantity)
        order_quantity = int(np.clip(action[0], 0, self.config['max_inventory']))
        
        # Record inventory before ordering
        inventory_before = np.sum(self.inventory)
        
        # Add new inventory with full shelf life
        self.inventory[-1] += order_quantity
        
        # Record inventory after ordering
        inventory_after_order = np.sum(self.inventory)
        
        # Calculate today's demand based on day of week
        day_factor = self.config['weekday_factors'][self.day_of_week]
        mean_demand = self.config['demand_mean'] * day_factor
        today_demand = max(0, int(self.np_random.normal(mean_demand, self.config['demand_std'])))
        
        # Update demand history
        self.demand_history = np.roll(self.demand_history, -1)
        self.demand_history[-1] = today_demand
        
        # Process demand fulfillment and calculate sales
        sales, stockouts = self._fulfill_demand(today_demand)
        
        # Age inventory and calculate waste
        waste = self._age_inventory()
        
        # Record inventory after demand
        inventory_after_demand = np.sum(self.inventory)
        
        # Calculate costs and profit
        purchase_cost = order_quantity * self.config['purchase_cost']
        holding_cost = inventory_after_demand * self.config['holding_cost']
        stockout_cost = stockouts * self.config['stockout_cost']
        waste_cost = waste * (self.config['purchase_cost'] + self.config['waste_cost'])  # Cost of the item plus disposal
        revenue = sales * self.config['selling_price']
        
        profit = revenue - (purchase_cost + holding_cost + stockout_cost + waste_cost)
        
        # Update tracking variables
        self.total_sales += sales
        self.total_waste += waste
        self.total_stockouts += stockouts
        
        # Record history for this step
        self._record_step_history(
            order_quantity, today_demand, inventory_before, 
            inventory_after_order, inventory_after_demand,
            sales, waste, stockouts, revenue, 
            purchase_cost, holding_cost, stockout_cost, waste_cost, profit
        )
        
        # Update day counter
        self.current_day += 1
        self.day_of_week = (self.day_of_week + 1) % 7
        
        # Check if episode is finished
        terminated = self.current_day >= self.config['episode_length']
        truncated = False
        
        # Get current observation
        observation = self._get_observation()
        
        # Create info dictionary
        info = {
            'sales': sales,
            'waste': waste,
            'stockouts': stockouts,
            'profit': profit,
            'total_inventory': np.sum(self.inventory)
        }
        
        return observation, profit, terminated, truncated, info
    
    def _get_observation(self):
        """Construct the observation from the current state."""
        return {
            'inventory': self.inventory,
            'day_of_week': np.array([self.day_of_week], dtype=np.int32),
            'demand_history': self.demand_history
        }
    
    def _fulfill_demand(self, demand):
        """
        Fulfill customer demand from inventory, starting with oldest inventory.
        
        Returns:
            tuple: (sales, stockouts)
        """
        sales = 0
        remaining_demand = demand
        
        # Use inventory to satisfy demand, starting with oldest inventory
        for i in range(self.config['shelf_life']):
            use_units = min(self.inventory[i], remaining_demand)
            self.inventory[i] -= use_units
            sales += use_units
            remaining_demand -= use_units
            
            if remaining_demand <= 0:
                break
        
        stockouts = remaining_demand  # Any demand not fulfilled results in stockouts
        return sales, stockouts
    
    def _age_inventory(self):
        """
        Age the inventory by one day and remove expired items.
        
        Returns:
            int: Number of wasted (expired) units
        """
        # Items at index 0 will expire
        waste = self.inventory[0]
        
        # Shift inventory (age by one day)
        self.inventory = np.roll(self.inventory, -1)
        
        # New items (ordered today) will be added at the last position
        # This has already been handled in the step method
        self.inventory[-1] = 0
        
        return int(waste)
    
    def _record_step_history(self, order_quantity, demand, inventory_before, 
                             inventory_after_order, inventory_after_demand,
                             sales, waste, stockouts, revenue, 
                             purchase_cost, holding_cost, stockout_cost, waste_cost, profit):
        """Record the history of this step for later analysis."""
        self.episode_history['day'].append(self.current_day)
        self.episode_history['day_of_week'].append(self.day_of_week)
        self.episode_history['demand'].append(demand)
        self.episode_history['order_quantity'].append(order_quantity)
        self.episode_history['inventory_before_order'].append(inventory_before)
        self.episode_history['inventory_after_order'].append(inventory_after_order)
        self.episode_history['inventory_after_demand'].append(inventory_after_demand)
        self.episode_history['sales'].append(sales)
        self.episode_history['waste'].append(waste)
        self.episode_history['stockouts'].append(stockouts)
        self.episode_history['revenue'].append(revenue)
        self.episode_history['purchase_cost'].append(purchase_cost)
        self.episode_history['holding_cost'].append(holding_cost)
        self.episode_history['stockout_cost'].append(stockout_cost)
        self.episode_history['waste_cost'].append(waste_cost)
        self.episode_history['profit'].append(profit)
        self.episode_history['total_inventory'].append(np.sum(self.inventory))
    
    def get_episode_history(self):
        """Return the history of this episode as a pandas DataFrame."""
        return pd.DataFrame(self.episode_history)
    
    def render(self, mode='human'):
        """Render the environment."""
        if mode != 'human':
            raise NotImplementedError(f"Render mode {mode} is not supported.")
        
        # Print current state
        print(f"Day: {self.current_day} (Weekday: {self.day_of_week})")
        print(f"Inventory: {self.inventory} (Total: {np.sum(self.inventory)})")
        print(f"Recent demand: {self.demand_history}")
        print(f"Total sales: {self.total_sales}, Total waste: {self.total_waste}, Total stockouts: {self.total_stockouts}")
    
    def close(self):
        """Clean up resources."""
        pass


class MultiProductSupermarketEnv(gym.Env):
    """
    Multi-Product Supermarket Inventory Management Environment
    
    This environment extends the basic SupermarketEnv to handle multiple products,
    each with their own characteristics (shelf life, cost, price, demand patterns).
    
    State:
        - Current inventory levels for each product (for each day of shelf life remaining)
        - Day of week (0-6)
        - Recent demand history for each product
        
    Action:
        - Order quantities for each product
        
    Reward:
        - Total profit across all products
    """
    
    metadata = {'render.modes': ['human']}

    def __init__(self, product_configs=None):
        """
        Initialize the multi-product environment.
        
        Args:
            product_configs (dict): Dictionary mapping product names to their configurations
        """
        if product_configs is None:
            # Default configuration for three products
            self.product_configs = {
                'fresh_produce': {
                    'max_inventory': 100,
                    'shelf_life': 4,  # Shorter shelf life
                    'purchase_cost': 1.2,
                    'selling_price': 3.0,
                    'holding_cost': 0.15,
                    'stockout_cost': 1.2,
                    'waste_cost': 0.8,
                    'demand_mean': 18,
                    'demand_std': 6,
                    'weekday_factors': [0.7, 0.7, 0.8, 0.9, 1.1, 1.6, 1.4]  # Higher weekend demand
                },
                'dairy': {
                    'max_inventory': 80,
                    'shelf_life': 7,  # Medium shelf life
                    'purchase_cost': 0.9,
                    'selling_price': 2.2,
                    'holding_cost': 0.1,
                    'stockout_cost': 1.0,
                    'waste_cost': 0.5,
                    'demand_mean': 15,
                    'demand_std': 4,
                    'weekday_factors': [0.8, 0.8, 0.9, 1.0, 1.1, 1.4, 1.2]  # Moderate weekend effect
                },
                'bakery': {
                    'max_inventory': 60,
                    'shelf_life': 3,  # Very short shelf life
                    'purchase_cost': 0.7,
                    'selling_price': 2.0,
                    'holding_cost': 0.08,
                    'stockout_cost': 0.8,
                    'waste_cost': 0.4,
                    'demand_mean': 22,
                    'demand_std': 7,
                    'weekday_factors': [0.9, 0.9, 1.0, 1.0, 1.1, 1.3, 1.2]  # Less weekend effect
                }
            }
        else:
            self.product_configs = product_configs
            
        # Common configuration across all products
        self.common_config = {
            'episode_length': 30,
            'seed': None
        }
        
        # Create individual environments for each product
        self.product_envs = {}
        for product_name, config in self.product_configs.items():
            # Merge common config with product-specific config
            full_config = self.common_config.copy()
            full_config.update(config)
            self.product_envs[product_name] = SupermarketEnv(full_config)
            
        # Product names in a stable order
        self.product_names = sorted(self.product_configs.keys())
        self.num_products = len(self.product_names)
        
        # Action space: order quantities for all products
        action_spaces = []
        for product_name in self.product_names:
            action_spaces.append(self.product_envs[product_name].action_space)
        self.action_space = spaces.Tuple(action_spaces)
        
        # Observation space: combined observations from all products + day of week
        # All products share the same day of week, so we only need one copy
        obs_spaces = {
            'day_of_week': spaces.Box(low=0, high=6, shape=(1,), dtype=np.int32)
        }
        
        for product_name in self.product_names:
            env = self.product_envs[product_name]
            obs_spaces[f'{product_name}_inventory'] = env.observation_space['inventory']
            obs_spaces[f'{product_name}_demand_history'] = env.observation_space['demand_history']
            
        self.observation_space = spaces.Dict(obs_spaces)
        
        # Initialize state
        self.reset()
        
    def reset(self, seed=None, options=None):
        """Reset all product environments."""
        # Reset each product environment
        for product_name in self.product_names:
            self.product_envs[product_name].reset(seed=seed)
            
        # Sync day of week across all environments
        self.day_of_week = 0
        for product_name in self.product_names:
            self.product_envs[product_name].day_of_week = self.day_of_week
            
        # Current day in episode
        self.current_day = 0
        
        # Get initial observation
        observation = self._get_observation()
        
        # Initialize history
        self.episode_history = {
            'day': [],
            'day_of_week': []
        }
        
        # Add product-specific metrics to history
        for product_name in self.product_names:
            self.episode_history[f'{product_name}_order'] = []
            self.episode_history[f'{product_name}_demand'] = []
            self.episode_history[f'{product_name}_inventory'] = []
            self.episode_history[f'{product_name}_sales'] = []
            self.episode_history[f'{product_name}_waste'] = []
            self.episode_history[f'{product_name}_stockouts'] = []
            self.episode_history[f'{product_name}_profit'] = []
            self.episode_history[f'{product_name}_avg_shelf_duration'] = []  # Add product-specific shelf duration
        
        self.episode_history['total_profit'] = []
        self.episode_history['total_waste'] = []
        self.episode_history['total_stockouts'] = []
        
        return observation, {}
        
    def step(self, action):
        """
        Take a step in all product environments.
        
        Args:
            action: Tuple of order quantities for each product
            
        Returns:
            observation: Combined observation
            reward: Total profit across all products
            terminated: Whether the episode is over
            truncated: Whether the episode was truncated
            info: Combined info dictionaries
        """
        # Process each product's action
        total_profit = 0
        total_waste = 0
        total_stockouts = 0
        infos = {}
        
        # Record day information
        self.episode_history['day'].append(self.current_day)
        self.episode_history['day_of_week'].append(self.day_of_week)
        
        # Step through each product environment
        for i, product_name in enumerate(self.product_names):
            env = self.product_envs[product_name]
            
            # Extract action for this product
            product_action = action[i].reshape(1)  # Reshape to match environment's expectation
            
            # Take step in this product's environment
            _, profit, terminated, truncated, info = env.step(product_action)
            
            # Accumulate results
            total_profit += profit
            total_waste += info['waste']
            total_stockouts += info['stockouts']
            
            # Record product-specific history
            self.episode_history[f'{product_name}_order'].append(product_action[0])
            self.episode_history[f'{product_name}_demand'].append(env.demand_history[-1])
            self.episode_history[f'{product_name}_inventory'].append(np.sum(env.inventory))
            self.episode_history[f'{product_name}_sales'].append(info['sales'])
            self.episode_history[f'{product_name}_waste'].append(info['waste'])
            self.episode_history[f'{product_name}_stockouts'].append(info['stockouts'])
            self.episode_history[f'{product_name}_profit'].append(profit)
            
            # Calculate and record product-specific average remaining shelf duration
            inv = env.inventory
            days = np.arange(len(inv))
            product_avg_remain = (days * inv).sum() / inv.sum() if inv.sum() > 0 else 0.0
            self.episode_history[f'{product_name}_avg_shelf_duration'].append(product_avg_remain)
            
            # Store info
            infos[product_name] = info
            
        # Record total metrics
        self.episode_history['total_profit'].append(total_profit)
        self.episode_history['total_waste'].append(total_waste)
        self.episode_history['total_stockouts'].append(total_stockouts)


        # Sync day of week across all environments
        self.day_of_week = (self.day_of_week + 1) % 7
        for product_name in self.product_names:
            self.product_envs[product_name].day_of_week = self.day_of_week
            
        # Update day counter
        self.current_day += 1
        
        # Check if episode is finished (use first product's termination condition)
        first_product = self.product_names[0]
        terminated = self.product_envs[first_product].current_day >= self.common_config['episode_length']
        truncated = False
        
        # Get combined observation
        observation = self._get_observation()
        
        # Add summary information to info
        infos['total_profit'] = total_profit
        infos['total_waste'] = total_waste
        infos['total_stockouts'] = total_stockouts
        
        return observation, total_profit, terminated, truncated, infos
    
    def _get_observation(self):
        """Combine observations from all product environments."""
        observation = {
            'day_of_week': np.array([self.day_of_week], dtype=np.int32)
        }
        
        for product_name in self.product_names:
            env = self.product_envs[product_name]
            env_obs = env._get_observation()
            observation[f'{product_name}_inventory'] = env_obs['inventory']
            observation[f'{product_name}_demand_history'] = env_obs['demand_history']
            
        return observation
    
    def get_episode_history(self):
        """Return the history of this episode as a pandas DataFrame."""
        return pd.DataFrame(self.episode_history)
    
    def render(self, mode='human'):
        """Render all product environments."""
        if mode != 'human':
            raise NotImplementedError(f"Render mode {mode} is not supported.")
            
        print(f"Day: {self.current_day} (Weekday: {self.day_of_week})")
        
        for product_name in self.product_names:
            env = self.product_envs[product_name]
            print(f"\n{product_name.upper()}")
            print(f"Inventory: {env.inventory} (Total: {np.sum(env.inventory)})")
            print(f"Recent demand: {env.demand_history}")
            print(f"Total sales: {env.total_sales}, Total waste: {env.total_waste}, Total stockouts: {env.total_stockouts}")
            
    def close(self):
        """Close all product environments."""
        for env in self.product_envs.values():
            env.close()
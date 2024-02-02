# Importing necessary libraries
import networkx as nx
from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.space import NetworkGrid
from mesa.datacollection import DataCollector
import geopandas as gpd
import rasterio as rs
import matplotlib.pyplot as plt
import random
import math
import os

# Import the agent class(es) from agents.py
from agents import Households, Government

# Import functions from functions.py
from functions import get_flood_map_data, calculate_basic_flood_damage,get_flood_depth
from functions import map_domain_gdf, floodplain_gdf


# Define the AdaptationModel class
class AdaptationModel(Model):
    """
    The main model running the simulation. It sets up the network of household agents,
    simulates their behavior, and collects data. The network type can be adjusted based on study requirements.
    """

    def __init__(self, 
                 seed = 123,
                 number_of_households = 50, # number of household agents, ##CAN WE REMOVE?
                 # Simplified argument for choosing flood map. Can currently be "harvey", "100yr", or "500yr".
                 flood_map_choice='100yr',
                 # ### network related parameters ###
                 # The social network structure that is used.
                 # Can currently be "erdos_renyi", "barabasi_albert", "watts_strogatz", or "no_network"
                 network = 'watts_strogatz',
                 # likeliness of edge being created between two nodes
                 probability_of_network_connection = 0.4,
                 # number of edges for BA network
                 number_of_edges = 3,
                 # number of nearest neighbours for WS social network
                 number_of_nearest_neighbours = 5,
                 a = 0.5,
                 b = 1.2,
                 APE = 0.01,
                 C = 0.01,
                 alpha = 1,
                 societal_risk = True
                 ):

        self.weak_adaptations = []
        self.medium_adaptations = []
        self.strong_adaptations = []
        
        super().__init__(seed = seed)
        
        # defining the variables and setting the values
        self.number_of_households = number_of_households  # Total number of household agents
        self.seed = seed
        self.a = a
        self.b = b
        self.APE = APE
        self.C = C
        self.alpha = alpha
        self.societal_risk = societal_risk

        # network
        self.network = network # Type of network to be created
        self.probability_of_network_connection = probability_of_network_connection
        self.number_of_edges = number_of_edges
        self.number_of_nearest_neighbours = number_of_nearest_neighbours

        # generating the graph according to the network used and the network parameters specified
        self.G = self.initialize_network()
        # create grid out of network graph
        self.grid = NetworkGrid(self.G)

        # Initialize maps
        self.initialize_maps(flood_map_choice)

        # set schedule for agents
        self.schedule = RandomActivation(self)  # Schedule for activating agents
        
        # create households through initiating a household on each node of the network graph
        for i, node in enumerate(self.G.nodes()):
            household = Households(unique_id=i, model=self, seed=self.seed)
            self.schedule.add(household)
            self.grid.place_agent(agent=household, node_id=node)

        # Create and add Government agent on the same node as the Household agent
            government_id = i + self.number_of_households  # Ensure unique ID for the Government agent
            government = Government(unique_id=government_id, model=self, location=household.location)
            self.schedule.add(government)
            self.grid.place_agent(agent=government, node_id=node)




        # Data collection setup to collect data
        model_metrics = {
                        "total_adapted_households": self.total_adapted_households,
                        # ... other reporters ...
                        }
        
        #For now these are collected, while actually we only need adaptation number and fatalities.
        agent_metrics = {
                        #"FloodDepthEstimated": "flood_depth_estimated",
                        #"FloodDamageEstimated" : "flood_damage_estimated",
                        #"FloodDepthActual": "flood_depth_actual",
                        #"FloodDamageActual" : "flood_damage_actual",
                        "DeathEstimated" : "death_population_est",
                        "DeathActual" : "death_population",
                        "IsAdaptedWeakly": "is_adapted_weak",
                        "IsAdaptedMedium": "is_adapted_medium",
                        "IsAdaptedStrongly": "is_adapted_strong",
                        #"Savings": "savings",
                        #"Is_insured": "is_insured",
                        #"FN_value_own": 'FN_value_own',
                        #"FN_standard_own": 'FN_standard_own',
                        #"location":"location",
                        # ... other reporters ...
                        }
        #set up the data collector 
        self.datacollector = DataCollector(model_reporters=model_metrics, agent_reporters=agent_metrics)

    def initialize_network(self):
        """
        Initialize and return the social network graph based on the provided network type using pattern matching.
        """
        if self.network == 'erdos_renyi':
            return nx.erdos_renyi_graph(n=self.number_of_households,
                                        p=self.number_of_nearest_neighbours / self.number_of_households,
                                        seed=self.seed)
        elif self.network == 'barabasi_albert':
            return nx.barabasi_albert_graph(n=self.number_of_households,
                                            m=self.number_of_edges,
                                            seed=self.seed)
        elif self.network == 'watts_strogatz':
            return nx.watts_strogatz_graph(n=self.number_of_households,
                                        k=self.number_of_nearest_neighbours,
                                        p=self.probability_of_network_connection,
                                        seed=self.seed)
        elif self.network == 'no_network':
            G = nx.Graph()
            G.add_nodes_from(range(self.number_of_households))
            return G
        else:
            raise ValueError(f"Unknown network type: '{self.network}'. "
                            f"Currently implemented network types are: "
                            f"'erdos_renyi', 'barabasi_albert', 'watts_strogatz', and 'no_network'")


    def initialize_maps(self, flood_map_choice):
        """
        Initialize and set up the flood map related data based on the provided flood map choice.
        """
        # Define paths to flood maps
        flood_map_paths = {
            'harvey': r'../input_data/floodmaps/Harvey_depth_meters.tif',
            '100yr': r'../input_data/floodmaps/100yr_storm_depth_meters.tif',
            '500yr': r'../input_data/floodmaps/500yr_storm_depth_meters.tif'  # Example path for 500yr flood map
        }

        # Throw a ValueError if the flood map choice is not in the dictionary
        if flood_map_choice not in flood_map_paths.keys():
            raise ValueError(f"Unknown flood map choice: '{flood_map_choice}'. "
                             f"Currently implemented choices are: {list(flood_map_paths.keys())}")

        # Choose the appropriate flood map based on the input choice
        flood_map_path = flood_map_paths[flood_map_choice]

        # Loading and setting up the flood map
        self.flood_map = rs.open(flood_map_path)
        self.band_flood_img, self.bound_left, self.bound_right, self.bound_top, self.bound_bottom = get_flood_map_data(
            self.flood_map)

    def total_adapted_households(self):
        """Return the total number of households that have adapted."""
        #BE CAREFUL THAT YOU MAY HAVE DIFFERENT AGENT TYPES SO YOU NEED TO FIRST CHECK IF THE AGENT IS ACTUALLY A HOUSEHOLD AGENT USING "ISINSTANCE"
        adapted_count = sum([1 for agent in self.schedule.agents if isinstance(agent, Households) and (agent.is_adapted_weak or agent.is_adapted_medium or agent.is_adapted_strong)])
        return adapted_count
    
    def plot_model_domain_with_agents(self, output_folder):
        fig, ax = plt.subplots()
        # Plot the model domain
        map_domain_gdf.plot(ax=ax, color='lightgrey')
        # Plot the floodplain
        floodplain_gdf.plot(ax=ax, color='lightblue', edgecolor='k', alpha=0.5)

        # Collect agent locations and statuses
        plot_list = [agent for agent in self.schedule.agents if isinstance(agent, Households)]
        for agent in plot_list:
            if agent.is_adapted_weak:
                color = 'yellow'
            elif agent.is_adapted_medium:
                color = 'green'
            elif agent.is_adapted_strong:
                color = 'blue'
            else:
                color = 'red'
            ax.scatter(agent.location.x, agent.location.y, color=color, s=10, label=color.capitalize() if not ax.collections else "")
            ax.annotate(str(agent.unique_id), (agent.location.x, agent.location.y), textcoords="offset points", xytext=(0,1), ha='center', fontsize=9)
        # Create legend with unique entries
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), title="Red: not adapted\nYellow: weakly adapted\nGreen: medium adapted\nBlue: strongly adapted", loc='upper left', bbox_to_anchor=(1, 1), ncol=1)

        # Customize plot with titles and labels
        plt.title(f'Model Domain with Agents at Step {self.schedule.steps}', y=1.05)
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')

        filename = f"model_domain_agents_step_{self.schedule.steps}.png"
        downloads_directory = os.path.expanduser("~/Downloads")
        output_folder = os.path.join(downloads_directory, "Images_ABM")
        filepath = os.path.join(output_folder, filename)
        os.makedirs(output_folder, exist_ok=True)
        plt.savefig(filepath)
        plt.show()
        plt.close()

    def step(self):
        """
        introducing a shock: 
        at time step 5, there will be a global flooding.
        This will result in actual flood depth. Here, we assume it is a random number
        between 0.5 and 1.2 of the estimated flood depth. In your model, you can replace this
        with a more sound procedure (e.g., you can devide the floop map into zones and 
        assume local flooding instead of global flooding). The actual flood depth can be 
        estimated differently
        """
        weak_count = sum(1 for agent in self.schedule.agents if isinstance(agent, Households) if agent.is_adapted_weak)
        medium_count = sum(1 for agent in self.schedule.agents if isinstance(agent, Households) if agent.is_adapted_medium)
        strong_count = sum(1 for agent in self.schedule.agents if isinstance(agent, Households) if agent.is_adapted_strong)

        self.weak_adaptations.append(weak_count)
        self.medium_adaptations.append(medium_count)
        self.strong_adaptations.append(strong_count)

        if self.schedule.steps == 0:
            households_list = [agent for agent in self.schedule.agents if isinstance(agent, Households)]
            list_of_deaths = []
            self.APE_in_own_node = self.APE
            self.affected_population_in_own_node = 94623
            self.exposure_rate_in_own_node = 0.1
            for agent in households_list:
                self.exposure_population = self.affected_population_in_own_node * self.exposure_rate_in_own_node
                if agent.flood_depth_estimated > 0:
                    self.death_rate_in_own_node_est = 0.655 * 0.001 * math.exp(1.16 * agent.flood_depth_estimated)
                else:
                    self.death_rate_in_own_node_est = 0
                agent.death_population_est = self.death_rate_in_own_node_est * self.exposure_population
                if agent.death_population_est > 94623:
                    agent.death_population_est = 94623
                list_of_deaths.append(agent.death_population_est)
                # Calculate the actual flood depth as a random number between 0.5 and 1.2 times the estimated flood depth
                agent.flood_depth_actual = random.uniform(a=self.a,b=self.b) * agent.flood_depth_estimated
                #print('estimated flood depth for households', agent.flood_depth_estimated)
                # calculate the actual flood damage given the actual flood depth
                # agent.flood_depth_actual = get_flood_depth(corresponding_map=self.flood_map, location=agent.location, band=self.band_flood_img)
                agent.flood_damage_actual = calculate_basic_flood_damage(agent.flood_depth_actual)
            self.total_population_death_est = sum(list_of_deaths)
            self.FN_standard_total = self.C / (self.total_population_death_est ** self.alpha)  ## For C, take 2.234 and for a use 0.703
            ## print('hell yeah!', int(self.total_population_death))

                # FN=1-Rn/(k+1) here Rn is the rank of the flood event, k is the event per year, we assume it is the APE
            self.FN_value_total = self.APE_in_own_node


        if self.schedule.steps == 49:
            households_list = []
            households_list = [agent for agent in self.schedule.agents if isinstance(agent, Households)]
            list_of_actual_deaths = []
            self.APE_in_own_node = self.APE
            self.affected_population_in_own_node = 94623
            self.exposure_rate_in_own_node = 0.1
            for agent in households_list:
                # agent.flood_depth_estimated = agent.flood_depth_actual
                self.exposure_population = self.affected_population_in_own_node * self.exposure_rate_in_own_node
                if agent.flood_depth_actual > 0:
                    self.death_rate_in_own_node = 0.655 * 0.001 * math.exp(1.16 * agent.flood_depth_actual)
                else:
                    self.death_rate_in_own_node = 0
                agent.death_population = self.death_rate_in_own_node * self.exposure_population
                if agent.death_population > 94623:
                    agent.death_population = 94623
                list_of_actual_deaths.append(agent.death_population)
                #print('deaths of the people are', agent.death_population)
                #print('death rate:', self.death_rate_in_own_node)
            self.total_population_death_actual = sum(list_of_actual_deaths)
            #self.FN_standard_actual = self.C / (self.total_population_death_actual ** self.alpha)
            #self.FN_value_actual = 1 - 1 / (self.APE_in_own_node + 1)
                # print('values are', self.FN_standard_actual,self.FN_value_actual)
                # if self.FN_value_actual < self.FN_standard_total:
                    # agent.is_adapted = True


        # Collect data and advance the model by one step
        self.datacollector.collect(self)
        self.schedule.step()

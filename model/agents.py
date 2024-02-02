# Importing necessary libraries
import math
import random
from mesa import Agent
from networkx import radius
from shapely.geometry import Point
from shapely import contains_xy
import numpy as np


# Import functions from functions.py
from functions import generate_random_location_within_map_domain, get_flood_depth, calculate_basic_flood_damage, floodplain_multipolygon


# Define the Households agent class
class Households(Agent):
    """
    An agent representing a household in the model.
    Each household has a flood depth attribute which is randomly assigned for demonstration purposes.
    In a real scenario, this would be based on actual geographical data or more complex logic.
    """

    def __init__(self, unique_id, model,seed):
        super().__init__(unique_id, model)
        self.is_adapted_weak = False
        self.is_adapted_medium = False
        self.is_adapted_strong = False
        # Initial adaptation status set to False.
        #self.spread_cal = None
        # make the householders in an initial saving group. We use a distribution of 0.2, 0.6, 0.2
        p_saving =[0.2, 0.6, 0.2]
        listsavings = ['low', 'middle' , 'high' ]
        self.savings = np.random.choice(listsavings, p=p_saving)
        
        # Initial state the household don't have insurance
        self.is_insured = False
        self.provide_insurance = False

        # house size of the householders, can be small, medium and large
        p_size=[0.2, 0.6, 0.2]
        listsize = ['small', 'medium' , 'large' ]
        self.housesize=np.random.choice(listsize, p=p_size )

        #household agent have a worry attribute
        self.worry= random.random()

        self.infra_applied = False
        self.subsidy_applied = False

        # getting flood map values
        # Get a random location on the map
        loc_x, loc_y = generate_random_location_within_map_domain(seed, unique_id)
        self.location = Point(loc_x, loc_y)

        # Check whether the location is within floodplain
        self.in_floodplain = False
        if contains_xy(geom=floodplain_multipolygon, x=self.location.x, y=self.location.y):
            self.in_floodplain = True
            #print(self.in_floodplain)

        # Get the estimated flood depth at those coordinates. 
        # the estimated flood depth is calculated based on the flood map (i.e., past data) so this is not the actual flood depth
        # Flood depth can be negative if the location is at a high elevation
        self.flood_depth_estimated = get_flood_depth(corresponding_map=model.flood_map, location=self.location, band=model.band_flood_img)
        #print(self.flood_depth_estimated)
        # handle negative values of flood depth
        if self.flood_depth_estimated < 0:
            self.flood_depth_estimated = 0
        #print('household estimated flood depth is',self.flood_depth_estimated, self.unique_id)

        # calculate the estimated flood damage given the estimated flood depth. Flood damage is a factor between 0 and 1
        self.flood_damage_estimated = calculate_basic_flood_damage(flood_depth=self.flood_depth_estimated)

        # Add an attribute for the actual flood depth. This is set to zero at the beginning of the simulation since there is not flood yet
        # and will update its value when there is a shock (i.e., actual flood). Shock happens at some point during the simulation
        self.flood_depth_actual = 0

        #calculate the actual flood damage given the actual flood depth. Flood damage is a factor between 0 and 1
        self.flood_damage_actual = calculate_basic_flood_damage(flood_depth=self.flood_depth_actual)

    def get_insurance(self):
        """This function is use to determine whether the household agent will get insurance or not"""
            #Logic to get insurance.
            #If someone provides insurance for the household, the household will get insurance under the following conditions:
            # He has to be in the flood plain
            # His worry have to be more than 0.6
            # The household have 1. small house, while he have a lot of savings. 2. Medium house at least medium savings 3. Large house
            #Now the insurance is cost 500, can be changed later
        if  self.in_floodplain == True and self.worry > 0.6 and self.provide_insurance == True and (self.savings == 'middle'or self.savings =='high'):
            self.is_insured = True

    
    # Functions for further evaluation of flood damage
    def damage_after_subsidy(self):
        """This function is used to change the households' estimation under multiple effects"""
        if (self.is_insured and self.is_adapted_medium) or (self.is_adapted_strong and self.no_high_dam_neighbours < 3):
            #print('before',self.flood_depth_actual)
            if  self.flood_depth_actual - (random.uniform(0.3, 0.75) * self.flood_depth_actual) >= 0:
                self.flood_depth_actual = self.flood_depth_actual - (random.uniform(0.3, 0.75) * self.flood_depth_actual)
            else:
                self.flood_depth_actual = 0
                #print("gained a subsidy")
            self.subsidy_applied = True
            return self.flood_depth_actual



    def worry_spread(self,radius):
        """This function is used to calculate the worry of a household agent"""
        troublemaker_list = []
        node_agents = self.model.grid.get_cell_list_contents([self.pos])
        spread_cal = 0
        government_effect = 0
        friend_list = self.model.grid.get_neighborhood(self.pos, include_center=False, radius=radius) ##Why radius is radius?
        id_list = [agent for agent in self.model.schedule.agents if isinstance(agent, Households)]

        for agent in node_agents:  # add communication by the government ## still no output here?
            if isinstance(agent, Government) and (
                    agent.weak_collective_adaptation == True and agent.medium_collective_adaptation == False and agent.strong_collective_adaptation == False):
                government_effect = 0.3
                self.is_adapted_weak = True ## Also possible
            elif isinstance(agent, Government) and (
                    agent.weak_collective_adaptation == False and agent.medium_collective_adaptation == True and agent.strong_collective_adaptation == False):
                government_effect = 0.6
                self.is_adapted_medium = True
            elif isinstance(agent, Government) and (
                    agent.weak_collective_adaptation == False and agent.medium_collective_adaptation == False and agent.strong_collective_adaptation == True):
                government_effect = 0.9
                self.is_adapted_strong = True
            else:
                government_effect = 0
            # print(government_effect)

        for agent in id_list:
            if agent.unique_id in friend_list:
                spread_cal = agent.worry + spread_cal
                # print('friend worry is', agent.worry)

        number_of_troublemaker = len(friend_list)
        if number_of_troublemaker < 1:
            number_of_troublemaker = 1

        self.worry = (self.worry + spread_cal) / (number_of_troublemaker + 1) + government_effect
        if self.worry > 1:
            self.worry = 1
        # print('my worry!', self.worry)
        return self.worry

    def insurance_decision(self):
        """Decide whether there is affordable insurance available or not"""
        # logic to provide insurance
        node_agents = self.model.grid.get_cell_list_contents([self.pos])
        for agent in node_agents:
            if isinstance(agent,Government) and (agent.no_adaptation == True or agent.weak_collective_adaptation == True or agent.medium_collective_adaptation == True):
                self.provide_insurance = True  # because we assume that before strong collective adaption, weak one will be done.
                #print('insurance provided')
            #else:
                #print('insurance not provided')

    def damage_after_infrastructure(self,radius):
        #if there are 3 or more than 3 highly damaged neighbours nearby, they get protected by infrastructure, which reduces flood depth randomly by 0.5 to 3 m
        neargov_list = self.model.grid.get_neighborhood(self.pos, include_center=False, radius=radius)
        high_damage_households = [agent for agent in self.model.schedule.agents if isinstance(agent, Households) and self.is_adapted_strong == True]
        high_damage_neighbours = []
        for agent in high_damage_households:
            if agent.unique_id in neargov_list:
                high_damage_neighbours.append(agent)
        #print('firstly',self.flood_depth_actual)
        self.no_high_dam_neighbours = len(high_damage_neighbours)
        #print("No of high dam neighbours", no_high_dam_neighbours)
        if self.no_high_dam_neighbours >= 3:
            self.flood_depth_actual = self.flood_depth_actual - random.uniform(0.5, 3)
            if self.flood_depth_actual <= 0:
                self.flood_depth_actual = 0
            self.infra_applied = True
        #print('secondly',self.flood_depth_actual)


    def step(self):
        
        #The step function of the household, doing these functions.
        self.worry_spread(radius=1)
        self.insurance_decision()
        self.get_insurance()
        if self.infra_applied == False:
            self.damage_after_infrastructure(radius=1)
        if self.subsidy_applied == False:
            self.damage_after_subsidy()




# Define the Government agent class
class Government(Agent):
    """
    A government agent will make policies regarding the adaptation according the society risk (FN curve and here we related it with flood depth).
    """
    def __init__(self, unique_id, model, location):
        super().__init__(unique_id, model)
        # You can find the seperate RBB for the collective adaptation policies.
        # For the government, for now it has three policies, weak and strong in general. This could be changed to more detailed strategies.
        # Weak = communication
        # Medium = commu + sub
        # strong = commu + infra
        self.weak_collective_adaptation = False
        self.medium_collective_adaptation = False
        self.strong_collective_adaptation = False
        self.no_adaptation = False

        self.location = location
        # Initial flood damage = 0 because flood happens at step 5
        self.flood_depth_actual = 0

        # calculate the actual flood damage given the actual flood depth. Flood damage is a factor between 0 and 1
        self.flood_damage_actual = calculate_basic_flood_damage(flood_depth=self.flood_depth_actual)
        
        #  One government agent on different node
        #  Can we use one government agent to evaluate all the places with household agents?
        ## one government agent at each dot, it is now not right.
        #self.location = Point(loc_x, loc_y)

        self.APE_in_own_node = model.APE  ##Changed this to 1/200
        self.affected_population_in_own_node = 94623
        self.exposure_rate_in_own_node = 0.1
        self.flood_depth_estimated = get_flood_depth(corresponding_map=model.flood_map, location=self.location,
                                                     band=model.band_flood_img)
        if self.flood_depth_estimated < 0:
            self.flood_depth_estimated = 0
        self.flood_damage_estimated = calculate_basic_flood_damage(flood_depth=self.flood_depth_estimated)
        #('government flood depth is', self.flood_depth_estimated, self.unique_id)


    def calculate_FNstandard(self):
        """
              This function is to calculate the FN curve.

              Parameters
              ----------
              APE: annual probability of exceedance(years), we don't have data, have to make assumptions. For 200 years, we assume it was 1/200, i.e. 0.5%
              affected_population: the number of population affected in the flooded area. This can be counted according to the network?
              exposure_rate: rate of affected_population who are exposured to the flood, we can make assumptions, 10% for instance.
              death_rate: rate of deaths who are exposured to the flood, we can make assumptions, 0.0325% for instance. This can be linked to damage factors.

              Returns
              -------
              FN_standard: return to the FN standard value with the corresponding death population
              """

        self.exposure_population = self.affected_population_in_own_node * self.exposure_rate_in_own_node
        if self.flood_depth_estimated > 0:
            self.death_rate_in_own_node = 0.655 * 0.001 * math.exp(1.16 * self.flood_depth_estimated)
        else:
            self.death_rate_in_own_node = 0
        self.death_population = self.death_rate_in_own_node * self.exposure_population
        # FN_standard= C/(death_population^alpha) we make assupmtions on C and alpha
        if self.death_population > 0:
            self.FN_standard_own = (self.model.C/50) / (self.death_population ** self.model.alpha) ## For C, take 2.234/50 and for a use 0.703? This was used in the paper. I also didn't understand why the result was times 1000?
        else:
            self.FN_standard_own = 1
        #print('the death rate is', self.death_population)
        return self.FN_standard_own



    def calculate_FNvalue(self):
        """
        This function is to calculate the FN value.
        Due to lot of uncertainties, we will make it as simple as possible
        Parameters
        ----------
        APE: annual probability of exceedance(years), we don't have data, have to make assumptions. For 200 years, we assume it was 1/200, i.e. 0.5%

        Returns
        -------
        FN_value: return to the FN value with the corresponding APE
        """
        # FN=1-Rn/(k+1) here Rn is the rank of the flood event, k is the event per year, we assume it is the APE.
        self.FN_value_own = self.APE_in_own_node

    #Functions to determine what to use    
    def step(self):
        # Logic for adaptation based on the FN value and FN standard
        # Standard should have two, to determine whether to use strong or weak adaptions
        ## This shouldn't be every step though right?

        if self.model.societal_risk == True:
            if self.model.FN_value_total > 0.5 * self.model.FN_standard_total:
                self.calculate_FNstandard()
                self.calculate_FNvalue()
                #print('the government takes action')
                #print(self.FN_value_own, self.FN_standard_own)

                if self.FN_value_own > 0.5 * self.FN_standard_own and self.FN_value_own <= 0.75 * self.FN_standard_own and random.random() > 0.7:
                    self.weak_collective_adaptation = True  # government agent decide on collective adaption strategies
                    # print('there is weak adaptation')
                if self.FN_value_own > 0.75 * self.FN_standard_own and self.FN_value_own <= self.FN_standard_own and random.random() > 0.85:
                    self.medium_collective_adaptation = True
                    # print('there is medium adaptation')
                if self.FN_value_own > self.FN_standard_own and random.random() > 0.95:
                    self.strong_collective_adaptation = True
                    # print('there is strong adaptation')
                else:
                    self.no_adaptation = True

            #if self.model.FN_value_total > 0.1 * self.model.FN_standard_total:
            elif self.model.FN_value_total > 0.25 * self.model.FN_standard_total and self.model.FN_value_total <= 0.5 * self.model.FN_standard_total:
                #print('take small route')
                self.calculate_FNstandard()
                self.calculate_FNvalue()
                #print(self.FN_value_own, self.FN_standard_own)
                # print('the government takes action')
                #print(self.FN_value_own, self.FN_standard_own)

                if self.FN_value_own > 0.75 * self.FN_standard_own and self.FN_value_own <= self.FN_standard_own and random.random() > 0.7:
                    self.weak_collective_adaptation = True  # government agent decide on collective adaption strategies
                    # print('there is weak adaptation')
                if self.FN_value_own > self.FN_standard_own and self.FN_value_own <= 1.25 * self.FN_standard_own and random.random() > 0.85:
                    self.medium_collective_adaptation = True
                    # print('there is medium adaptation')
                if self.FN_value_own > 1.25 * self.FN_standard_own and random.random() > 0.95:
                    self.strong_collective_adaptation = True
                    # print('there is strong adaptation')
                else:
                    self.no_adaptation = True

        elif self.model.societal_risk == False:
            self.rnr = 10
            self.calculate_FNstandard()
            self.calculate_FNvalue()
            #print(self.FN_value_own, self.FN_standard_own)
            if self.FN_value_own > self.rnr * 0.5 * self.FN_standard_own and self.FN_value_own <= self.rnr * 0.75 * self.FN_standard_own and random.random() > 0.7:
                self.weak_collective_adaptation = True  # government agent decide on collective adaption strategies
                # print('there is weak adaptation')
            if self.FN_value_own > self.rnr * 0.75 * self.FN_standard_own and self.FN_value_own <= self.rnr * self.FN_standard_own and random.random() > 0.8:
                self.medium_collective_adaptation = True
                # print('there is medium adaptation')
            if self.FN_value_own > self.rnr * self.FN_standard_own and random.random() > 0.9:
                self.strong_collective_adaptation = True
                # print('there is strong adaptation')
            else:
                self.no_adaptation = True








        
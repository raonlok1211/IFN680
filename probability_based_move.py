
#----- IFN680 Assignment 1 -----------------------------------------------#
#  The Wumpus World: a probability based agent
#
#  Implementation of two functions
#   1. PitWumpus_probability_distribution()
#   2. next_room_prob()
#
#    Student no: n7588844 
#                n10328611
#    Student name: Tzu-Wen CHEN (Vincent)
#                  Earl Yin Lok CHAU 
#
#-------------------------------------------------------------------------#
from random import *
from AIMA.logic import *
from AIMA.utils import *
from AIMA.probability import *
from tkinter import messagebox

import logic_based_move
#--------------------------------------------------------------------------------------------------------------
#
#  The following two functions are to be developed by you. They are functions in class Robot. If you need,
#  you can add more functions in this file. In this case, you need to link these functions at the beginning
#  of class Robot in the main program file the_wumpus_world.py.
#
#--------------------------------------------------------------------------------------------------------------

# =============================================================================
# This function is to create an object of JointProbDist to store the joint probability distribution and  
# return the object. The object will be used in the function next_room_prob() # to calculate the 
# required probabilities.
# 
# This function will be called in the constructor of class Robot in the main program the_wumpus_world.py 
# to construct the joint probability distribution object. 
# 
# This function will be called in the next_room_prob() to use the joint probability distribution
# to calculate the required conditional probabilities. 
# =============================================================================
def PitWumpus_probability_distribution(self, width, height): 
    # Create a list of variable names to represent the rooms. 
    # A string '(i,j)' is used as a variable name to represent a room at (i, j)
    self.PW_variables = [] 
    for column in range(1, width + 1):
        for row in range(1, height + 1):
            self.PW_variables  = self.PW_variables  + ['(%d,%d)'%(column,row)]

    #--------- Add your code here -------------------------------------------------------------------

    # Define two constants to represent the prior probability of the pitwrump occurance for each room
    p_true = 0.2 # rooms that contain pi/wrump
    p_false = 1 - 0.2 #every rooms that not containing pit/wrump 
     
    #Each variable has two values, True or False, Create a dict to specify the value domain for each variable. 
    var_values = {each: [T, F] for each in self.PW_variables}
 
    #Create an object of JointProbDist to represent the joint probability distribution for the N variables
    Pr_each_event = JointProbDist(self.PW_variables, var_values)

    #Generate all events
    events = all_events_jpd(self.PW_variables, Pr_each_event, {})    
    
    #Assign a probability to each of the events
    for each_event in events:
        # Calculate the probability for this event
        # if the value of a variable is false, motiply by p_false which is 0.2, otherwise motiply by p_true which is 1-0.2 
        prob = 1 # initial value of the probability
        for (var, val) in each_event.items(): # for each (variable, value) pair in the dictionary
            prob = prob * p_false if val == F else prob * p_true
        # Assign the probability to this event        
        Pr_each_event[each_event] = prob

    #check if the sum of all probabily is 1
    if Pr_each_event.is_valid():
        print (" is valid.\n")
    else: print (" is invalid.\n")
    return Pr_each_event
            
#---------------------------------------------------------------------------------------------------
# =============================================================================
# #   Function 2. next_room_prob(self, x, y)
# #
# #  The parameters, (x, y), are the robot's current position in the cave environment.
# #  x: column
# #  y: row
# #
# #  This function returns a room location (column,row) for the robot to go.
# #  There are three cases: 
# #
# #    Case 1 : If the robot found a safe room, use the logic base move function,
# #             Otherwise, use probability based move function
# #    
# #    Case 2 : If the robot cannot found a safe room, the robot will choose a room 
# #             whose probability of containing a pit/wumpus is lower than the
# #             pre-specified probability threshold, and the lowest probability
# #             amount the query rooms, then return the location of that room.
# #
# #    Case 3 : If the probabilities of all the surrounding rooms are not lower than 
# #             the pre-specified probability threshold, return (0,0).
# =============================================================================

def next_room_prob(self, x, y):
    R_known = {}    #set
    R_query = []    #list
    R_other = []    #list              
    BS_known = dict()   #Dict
    PW_known = dict()   #Dict
    
    print("moves",self.num_moves)
    print("path",self.path)
    print("visited rooms",self.visited_rooms)
    print("available rooms",self.available_rooms)
    print("==========================")
    
    #if logic based can find a safe room, use the logic based move
    if (logic_based_move.next_room(self,x,y) != (0,0)):
        return logic_based_move.next_room(self,x,y) 
    #otherwise,logic based cant find a 100% safe room, use probability based move
    else:     
    #R_known is a list of rooms that the agent has visited 
        R_known = self.visited_rooms      
        
    #R_query is a list of rooms that adjacent to hte agent's current location and have not been visited                                  
        R_query = self.cave.getsurrounding(x,y)  
        print("R_query",R_query)
        for room in self.visited_rooms:
            if room in R_query:
                R_query.remove(room)
        print("R_query removed visited",R_query)
        
    #R_others is a list of rooms that agent has not visited and not adjacent to the agent's current location
        #Get all rooms and store in list() R_others
        for column in range(1, self.cave.WIDTH + 1):
            for row in range(1, self.cave.HEIGHT + 1):
                R_other  = R_other  + [(column,row)]             
      
        #updating R_other list(), remove visited room 
        for room in self.visited_rooms:
            if room in R_other:
                R_other.remove(room)
                
        #remove Room that are adjacent to agent location
        for room in R_query:
            if room in R_other:
                R_other.remove(room)

    #based on the R_query list, calculate the indivioual query room prob Pq true      
        lowest_pr = 1 #declare the lowest_pr is 1 for finding the lowest pq rooom pr
        lowest_pr_room = () #declare a vaviable store the lowest pr rpp,
        
        #get the visited room's information
        BS_known = self.observation_breeze_stench(self.visited_rooms) #dict
        PW_known = self.observation_pits(self.visited_rooms) #dict
        
        #start calculating the Pq true
        for room in R_query:                   
            print("Current Query Room",room)
        #create a set() R_unknown, which  R_unkown = R_others U (R_query-Pq)
        #Finding R_others U (R_query-Pq)     
            R_unknown = set()                                
            
            R_unknown.update(R_other) #update the set() R_unknow equal to R_others                 
            
            temp = []
            temp = R_query.copy()
            temp.remove(room)
            R_unknown.update(temp)  #update the set() R_unknow equal to (R_query-Pq)
            
            #change R_unknown from set to string, as for all_events_jpd
            R_unknown_str = []
            for each in R_unknown:
                R_unknown_str.append('({},{})'.format(each[0],each[1]))
            
        #create all unknown events assuming current query room PW = True 
            #assuming the current query room have pit/wump
            PW_known['({},{})'.format(room[0],room[1])] = True
            All_possible_events = all_events_jpd(R_unknown_str,self.jdP_PWs,PW_known)
            
        #calculate the probibilty of current query room that have Pit/Wump, Pq =true
            pqt = 0
            for event in All_possible_events:
                pqt = pqt + self.jdP_PWs[event]*self.consistent(BS_known, event)
        
        #create all unknown events assuming current query room PW = False
            #assuming the current query room dont have pit/wump
            PW_known['({},{})'.format(room[0],room[1])] = False
            All_possible_events = all_events_jpd(R_unknown_str,self.jdP_PWs,PW_known) 
 
        #calculate the probibilty of current query room that have Pit/Wump, Pq = false           
            pqf = 0
            for event in All_possible_events:
                pqf = pqf + self.jdP_PWs[event]*self.consistent(BS_known, event)
            
        #calculate the normalization of the Pq = True probability, beacuse some of the event removed and the total probability is not 1.
            normalize_pr = pqt / (pqt + pqf)
            print("normal_pr",normalize_pr)
          
        #find the lowest probability room within the R_query and store as lowest_pr, and stor the lowerst probability room as lowerst_pr_room         
            if(lowest_pr > normalize_pr ):
                lowest_pr = normalize_pr
                lowest_pr_room = room

    #Agent's moving decisions
        #if the all query rooms proabability are equual or lower than max_pit therhold, it is safe, return the lowest probability query room,lowest_pr_room    
        if(lowest_pr <= self.max_pit_probability): 
            print("go to ",lowest_pr_room)
            return lowest_pr_room   

        #otherwise all the query room is larger than max, it is unsafe, return (0,0)
        else:
            print("go back 1 move")
            return (0,0)  

        

#---------------------------------------------------------------------------------------------------

####################################################################################################

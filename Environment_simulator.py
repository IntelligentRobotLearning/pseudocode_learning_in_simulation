"""Interface demonstration of environment simulator"""

class Environment(object):

    def __init__(self,):
        pass

    def GetMuscleTuples(self):
        """Returns the hidden state of the agent for the start of an episode."""
        # Network details elided.
        return initial_state


    def GetExoObservations(self):
        """return the state of the exskeletion in the simulator"""
        return exo_state

    def GetHumanObservations(self):
        """return the state of the human in the simulator"""
        return human_state

    def SetHumanActions(self, actions):
        """Set the action for the human"""
        return None

    def SetExoActions(self, actions):
        """Set the action for the exskeletion"""
        return None

    def GetMuscleTorques(self):
        """return the muscle torque"""
        return muscle_torque

    def GetDesiredTorquesHuman(self):
        """return the desired torque of the human as the target"""
        return desired_torque_human
    
    def SetActivationLevels(self, activations):
        """set the activate levels of muscles in the simulator"""
        return None
    
    def Steps(self):
        """Performs simulation step in the simulator"""
        return None

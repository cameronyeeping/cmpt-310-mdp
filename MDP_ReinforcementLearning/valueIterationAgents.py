# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).
import queue

# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util
from queue import PriorityQueue

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def getGreedyUpdate(self, state):
        """computes a one step-ahead value update and return it"""
        if self.mdp.isTerminal(state):
            return self.values[state]
        actions = self.mdp.getPossibleActions(state)
        vals = util.Counter()
        for action in actions:
            vals[action] = self.computeQValueFromValues(state, action)
        return max(vals.values())

    def runValueIteration(self):
        # Write value iteration code here
        """
        V_k+1(s) = max(sum(T(s,a,s')[R(s,a,s')+ discount * V_k(s')]
        in other words, use previous iteration to compute current iteration
        V_k(s') is previous value iteration on s'
        - store value in self.values? (ie. self.values[state] = V_k(s')

        compute V_0, then V_1, then V_2, ...
        """
        states = self.mdp.getStates()
        new_values = self.values.copy()
        print('runValueIteration called')
        for _ in range(self.iterations):
            # DO NOT UPDATE SINGLE BATCH VECTOR IN PLACE
            new_values = self.values.copy()
            for state in states:
                if self.mdp.isTerminal(state):
                    new_values[state] = 0
                else:
                    actions = self.mdp.getPossibleActions(state)
                    # new_values[state] = max([self.getQValue(state, action) for action in actions])
                    new_values[state] = self.getQValue(state, self.getAction(state))
            self.values = new_values.copy()



    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    """ 
    V*(s) = max(sum(T(s,a,s')[R(s,a,s')+ d * V*(s')]))
    T is the transition function, R is the reward function, d is the discount
    V*(s) returns the EXPECTED utility starting in s, ending optimally

    function is recursive
    """

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.

          T(s,a,s') is probability of result state (s') occurring given state (s), action (s)
        """
        # value function stored in self.values

        # mdp.getReward(state, action, nextState)
        # list of (resultState, prob)
        q_value = 0
        possible = self.mdp.getTransitionStatesAndProbs(state, action)
        for nextState, prob in possible:
            q_value += prob * (self.mdp.getReward(state, action, nextState) + self.discount * self.values[nextState])
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.

        """
        # best action given current state = the result state with highest q value?

        actions = self.mdp.getPossibleActions(state)
        best_action = None
        if not actions:
            return best_action  # no legal actions; return None

        best_action = max(actions, key=lambda action: self.getQValue(state, action))

        return best_action  # return the action with the highest q value

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        # "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class PrioritizedSweepingValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take a mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def setupAllPredecessors(self):
        # compute predecessors of all states and save it in a util.Counter() and return it
        # what are predecessors of all states?
        # for each state, look at possible actions, and add state as predecessor to result state
        preds = collections.defaultdict(set)
        states = self.mdp.getStates()
        for state in states:
            preds[state] = set()
        for state in states:
            if not self.mdp.isTerminal(state):
                actions = self.mdp.getPossibleActions(state)
                for action in actions:
                    possible = self.mdp.getTransitionStatesAndProbs(state, action)
                    for result, _ in possible:
                        preds[result].add(state)


        return preds

    def setupPriorityQueue(self):
        """
          setup priority queue for all states based on their highest diff in greedy update
          - create a priority queue
        """
        pq = util.PriorityQueue()
        states = self.mdp.getStates()
        for state in states:
            if self.mdp.isTerminal(state):
                diff = self.values[state]
            else:
                diff = abs(self.values[state] - self.getQValue(state, self.getAction(state)))
            print(diff)
            pq.push(state, -diff)

        return pq

    def runValueIteration(self):
        # compute predecessors of all states
        # use a priority queue to determine which states are likely to benefit from an update
        allpreds = self.setupAllPredecessors()
        # print(allpreds)

        # setup priority queue
        pq = self.setupPriorityQueue()
        # run priority sweeping value iteration:
        for i in range(self.iterations):
            if pq.isEmpty():
                return
            state = pq.pop()
            if not self.mdp.isTerminal(state):
                self.values[state] = self.getQValue(state, self.getAction(state))
            for pred in allpreds[state]:
                diff = abs(self.values[pred] - self.getQValue(pred, self.getAction(pred)))
                if diff > self.theta:
                    pq.update(pred, -diff)







import random

import gtpyhop
import gymnasium as gym

from taxi_env import NavigateTaxiEnv


###############################################################################
# set Domain
domain_name = __name__
the_domain = gtpyhop.Domain(domain_name)


###############################################################################
# rigid relations, states, goals


# rigid relations
rigid = gtpyhop.State('rigid relations')
rigid.types = {
    'passenger': ['passenger_1'],
    'taxi': ['taxi_1'],
    'coords': [(r, c) for r in range(5) for c in range(5)],
    'pass_locs': ['0', '1', '2', '3', '4'],
    'color_squares': ['0', '1', '2', '3']
}


###############################################################################
# Helper functions:

def color_ind_to_coord(color_i):
    ci2c = {
        '0': (0, 0),
        '1': (0, 4),
        '2': (4, 0),
        '3': (4, 3)
    }
    return ci2c[color_i]


def coord_to_color_ind(coord):
    c2ci = {
        (0, 0): '0',
        (0, 4): '1',
        (4, 0): '2',
        (4, 3): '3'
    }
    return c2ci.get(coord, -1)


def is_a(variable, type):
    """
    from simple_hgn.py in GTPyhop's examples

    In most classical planners, one would declare data-types for the parameters
    of each action, and the data-type checks would be done by the planner.
    GTPyhop doesn't have a way to do that, so the 'is_a' function gives us a
    way to do it in the preconditions of each action, command, and method.

    'is_a' doesn't implement subtypes (e.g., if rigid.type[x] = y and
    rigid.type[x] = z, it doesn't infer that rigid.type[x] = z. It wouldn't be
    hard to implement this, but it isn't needed in the simple-travel domain.
    """
    return variable in rigid.types[type]


###############################################################################
# Actions:
def pick_up(state, p, pass_taxi_loc):
    state.loc[p] = pass_taxi_loc
    return state


def drop_off(state, p, gs):
    state.loc[p] = gs
    return state


def navigate(state, t, coord):
    state.loc[t] = coord
    return state


gtpyhop.declare_actions(pick_up, drop_off, navigate)


###############################################################################
# Helper functions for methods:
def get_taxi():
    t = rigid.types['taxi'][0]
    return t


###############################################################################
# Methods:
def m_pick_up(state, p, pass_taxi_loc):
    if is_a(p, 'passenger') and pass_taxi_loc == '4':
        t = get_taxi()
        if state.loc[t] == color_ind_to_coord(state.loc[p]):
            return [('pick_up', p, pass_taxi_loc)]


def m_drop_off(state, p, gs):
    if is_a(p, 'passenger') and is_a(gs, 'color_squares') and state.loc[p] == '4':
        t = get_taxi()
        t_color_ind = coord_to_color_ind(state.loc[t])
        if t_color_ind == gs:
            return [('drop_off', p, gs)]


def m_navigate(state, t, coord):
    if is_a(t, 'taxi') and is_a(coord, 'coords'):
        return [('navigate', t, coord)]


def m_full_service(state, p, gs):
    if is_a(p, 'passenger') and is_a(gs, 'color_squares'):
        t = get_taxi()
        p_loc = state.loc[p]
        if t:
            return [('loc', t, color_ind_to_coord(p_loc)),
                    ('loc', p, '4'),
                    ('loc', t, color_ind_to_coord(gs)),
                    ('loc', p, gs)]


gtpyhop.declare_unigoal_methods('loc', m_pick_up, m_drop_off, m_navigate, m_full_service)


###############################################################################
# Run the problem

# initial state
state0 = gtpyhop.State('state0')
state0.loc = {
    'taxi_1': (4, 4),
    'passenger_1': '1',
}

gtpyhop.find_plan(state0, [('loc', 'passenger_1', '0')])
gtpyhop.find_plan(state0, [('loc', 'passenger_1', '1')])
gtpyhop.find_plan(state0, [('loc', 'passenger_1', '2')])
gtpyhop.find_plan(state0, [('loc', 'passenger_1', '3')])
# gtpyhop.run_lazy_lookahead(state0, ['loc', 'passenger_1', 1])

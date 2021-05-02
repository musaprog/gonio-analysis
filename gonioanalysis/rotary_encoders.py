'''
Transforming rotary encoder step values into degrees and vice versa.
'''

DEFAULT_STEPS_PER_REVOLUTION = 1024 


def to_degrees(angle_pairs):
    '''
    Transform angle pairs (paired the steps of rotary encoder)
    to corresponding degree angle values in place.

    angle_pairs     List of (horizontal, vertical)
    '''

    for i in range(len(angle_pairs)):
        angle_pairs[i][0] *= (360/1024)
        angle_pairs[i][1] *= (360/1024)


def step2degree(step, steps_per_revolution=DEFAULT_STEPS_PER_REVOLUTION):
    '''
    Transform a rotary encoder step count (such 54 steps) into corresponding
    rotation in degrees (54 * steps_per_revolution).
    '''
    return step * (360/steps_per_revolution)


def degree2step(angle, steps_per_revolution=DEFAULT_STEPS_PER_REVOLUTION):
    '''
    Transform a rotation angle from degrees to corresponging rotary encoder steps.
    '''
    return angle * (steps_per_revolution/360)



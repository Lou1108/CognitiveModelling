
def start():
    return 1

def perceptualstep():
    return 1

def cognitivestep():
    return 1

def motorstep():
    return 1


def example1():
    """

    :return: total time of all cognitive processes
    """
    start_time = start()
    perceptual_time = perceptualstep()
    cog_time = cognitivestep()
    motor_time = motorstep()

    return start_time + perceptual_time + cog_time + motor_time
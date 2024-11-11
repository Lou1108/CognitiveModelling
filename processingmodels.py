import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def start(type="middle"):
    return 0


def perceptualstep(type="middle"):
    if type == "middle":
        return 100
    elif type == "fast":
        return 50
    else:
        return 200


def cognitivestep(type="middle"):
    if type == "middle":
        return 70
    elif type == "fast":
        return 25
    else:
        return 170


def motorstep(type="middle"):
    if type == "middle":
        return 70
    elif type == "fast":
        return 30
    else:
        return 100


def gen_box_whisker(data):
    fig = plt.figure(figsize=(10, 7))
    plt.boxplot(data)
    plt.show()


def gen_scatter_plot(data):
    fig = plt.figure(figsize=(10, 7))
    sns.scatterplot(data=data, x="time", y="error")
    fig.show()


def example1():
    """

    :return: total time of all cognitive processes
    """
    start_time = start()
    perceptual_time = perceptualstep()
    cog_time = cognitivestep()
    motor_time = motorstep()

    return start_time + perceptual_time + cog_time + motor_time


def call_all(type_p, type_c, type_m):
    start_time = start()
    perceptual_time = perceptualstep(type_p)
    cog_time = cognitivestep(type_c)
    motor_time = motorstep(type_m)

    return start_time + perceptual_time + cog_time + motor_time


def call_all_ideal(type_p, type_c, type_m):
    """
    call for souble stimulus
    :param type_p:
    :param type_c:
    :param type_m:
    :return:
    """
    start_time = start()
    perceptual_time = perceptualstep(type_p)
    cog_time = cognitivestep(type_c)
    motor_time = motorstep(type_m)

    return start_time + 2 * perceptual_time + 2 * cog_time + motor_time


def error_prob(type):
    if type == "fast":
        return 3
    elif type == "middle":
        return 2
    else:
        return 0.5


def call_all_ideal_error(type_p, type_c, type_m):
    """
    call for souble stimulus
    :param type_p:
    :param type_c:
    :param type_m:
    :return:
    """
    err_prob = 0.01
    e_p = error_prob(type_p)
    e_c = error_prob(type_c)
    e_m = error_prob(type_m)
    total_error = err_prob * e_p * e_p * e_c * e_c * e_m

    if total_error > 1:
        total_error = 1

    start_time = start()
    perceptual_time = perceptualstep(type_p)
    cog_time = cognitivestep(type_c)
    motor_time = motorstep(type_m)

    return start_time + 2 * perceptual_time + 2 * cog_time + motor_time, total_error


def example2(completeness="extremes"):
    """

    :param completeness: "extremes", "all"
    :return:
    """

    if completeness == "extremes":
        return call_all("fast", "fast", "fast"), call_all("middle", "middle", "middle"), call_all("slow", "slow", "slow")
    else:
        outcomes = []
        types = ["fast", "middle", "slow"]

        for s in types:
            for t in types:
                for u in types:
                    outcomes.append(call_all(s, t, u))

        return outcomes


def example3(completeness="extremes"):
    """

    :param completeness: "extremes", "all"
    :return:
    """

    if completeness == "extremes":
        return call_all_ideal("fast", "fast", "fast"), call_all_ideal("middle", "middle", "middle"), call_all_ideal("slow", "slow", "slow")
    elif completeness=="task":
        return call_all_ideal("fast", "middle", "slow")
    else:
        outcomes = []
        types = ["fast", "middle", "slow"]

        for s in types:
            for t in types:
                for u in types:
                    outcomes.append(call_all_ideal(s, t, u))

        return outcomes


def example4(completeness="extremes"):
    """

    :param completeness: "extremes", "all"
    :return:
    """

    second_stim = [40, 80, 110, 150, 210, 240]

    times = []
    for s in second_stim:
        results = example3(completeness)
        for r in results:
            times.append(r + s)
    return times


def example5(completeness="extremes"):
    """

    :param completeness: "extremes", "all"
    :return:
    """

    if completeness == "extremes":
        return call_all_ideal_error("fast", "fast", "fast"),\
            call_all_ideal_error("middle", "middle", "middle"), \
            call_all_ideal_error("slow", "slow", "slow")
    else:
        outcomes = []
        types = ["fast", "middle", "slow"]

        for s in types:
            for t in types:
                for u in types:
                    outcomes.append(call_all_ideal_error(s, t, u))

        return outcomes


out = example5("all")
print(out)
df = pd.DataFrame(out, columns=['time', 'error'])
gen_scatter_plot(df)
#gen_box_whisker(out)
import numpy as np


def duplicate_padding(x, length):
    """
    padding the x not with zeros but the copy of the x.

    :param: x: np.array or a list of np.array
        more general, it should be a list of objects, which has length and can be concatenate.
    :param: length: int
    :return np.array
    """

    x_new = np.zeros((len(x), length))
    for i, xx in enumerate(x):
        if len(xx) >= length:
            x_new[i, :] = xx[0: length]
        else:
            xx_copy_section = xx[0: (length - len(xx))]
            xx_replay = np.hstack((xx, xx_copy_section))  # np.concatenate()

            # concatenate copied x to original x until its length meets the upper bound
            while len(xx_replay) < length:
                xx_copy_section = xx[0:(length - len(xx_replay))]
                xx_replay = np.hstack((xx_replay, xx_copy_section))

            x_new[i, :] = xx_replay
    return x_new

import numpy as np


# increases argument in index position by err, to use in functional approach
def increase_arg_err(err, index, *args):
    new_args = np.array(args)
    new_args[index] = new_args[index] + err
    return tuple(new_args)


# functional approach for 1d function ( eg. y = x + 1 --> 1 error: x), return error
def functional_approach(func, err, *data, index=0):
    new_data = increase_arg_err(err, index, *data)
    return abs(func(*new_data) - func(*data))


# functional approach for nd function (eg. t = z * y + x ** 2 + 1 --> 3 errors: x,y,z), return error
def functional_approach_nd(func, *data_err_pairs):
    arrs, errs = [], []

    # data and errors must be provided in pairs of (values, errors), can mix & match arrays and single numbers
    for pair in data_err_pairs:
        arrs.append(pair[0])
        errs.append(pair[1])

    i_errs = [functional_approach(func, errs[i], *arrs, index=i) for i in range(len(arrs))]

    return np.sqrt(np.sum(np.array(i_errs) ** 2, axis=0))
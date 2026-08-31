import math


def get_intermediate_size(layer_id):
    #intermediate_size = int(14996 - 600 * layer_id)
    #intermediate_size = int(18238.89 * math.pow(0.92, layer_id))
    #intermediate_size = int(18239.5 * math.pow(0.92, layer_id))

    intermediate_size = int(20004 * math.pow(0.905, layer_id))
    #print(f'intermediate_size:{intermediate_size}')
    return intermediate_size

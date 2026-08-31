import math
d_min = 4096
d_max = 19792
#gamma = 0.7574
gamma=0.8155
delta = d_max - d_min

def get_intermediate_size(layer_id):
    osc = 0.5 * delta * (1 + math.cos(math.pi * layer_id / 15))
    raw_val = int(d_min + osc * gamma)
    #print(f'raw_val:{raw_val}')
    return int(raw_val)
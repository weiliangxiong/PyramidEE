

def get_intermediate_size(layer_id):
    #intermediate_size = int(14996 - 600 * layer_id)
    intermediate_size=10240+256
    #print(f'intermediate_size:{intermediate_size}')
    return intermediate_size
import numpy as np

def make_phi_range(nbins):
    low_phi = -np.pi
    step = 2*np.pi/nbins
    phi_range = [low_phi+x*step for x in range(nbins+1)]
    return phi_range

if __name__ == "__main__":
    print(make_phi_range(11))

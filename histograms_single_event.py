import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from utils.histograms import make_phi_range

def delta_phi(phi1, phi2):
    result = phi1 - phi2;
    while (result > np.pi):   result -= 2*np.pi
    while (result <= -np.pi): result += 2*np.pi
    return result

def vdelta_phi(x1, x2):
    return np.vectorize(delta_phi)(x1, x2)


# PARAMETERS
input_path  = './real.csv'
output_path = '/data/track-ml/output/'
plt.ioff()

# Open the input file
tracks = np.genfromtxt(input_path,delimiter=",")
print(tracks.shape)

# Slice the input file in its constituent parts
indexes = tracks[:,0]    # Particle indexes (0,1,2,...)
vertices = tracks[:,1:4] # Particle vertex (tx,ty,tz)
momenta = tracks[:,4:7]  # Particle momentum (px,py,pz)
hits = tracks[:,7:-1]    # N hits with the following information: 3D point + 3 tech. info
print(hits.shape)

global_X = np.empty(0)
global_Y = np.empty(0)
global_eta = np.empty(0)
global_phi = np.empty(0)
global_centered_eta = np.empty(0)
global_centered_phi = np.empty(0)
num_hits = []
eta_bins = [x / 10.0 for x in range(-50, 51, 1)]
phi_bins = [x / 10.0 for x in range(-32, 33, 1)]
delta_bins = [x / 10.0 for x in range(-20, 21, 1)]

# Loop over particles
for i in range(hits.shape[0]):
#for i in range(100):
    if(i%100==0):print(i)

    the_hits = hits[i]
    reshaped_hits = the_hits.reshape(int(114/6),6)
    X = reshaped_hits[:,0]
    Y = reshaped_hits[:,1]
    Z = reshaped_hits[:,2]
    X = np.trim_zeros(X)
    Y = np.trim_zeros(Y)
    Z = np.trim_zeros(Z)
    assert (X.size == Y.size and X.size == Z.size and Y.size == Z.size)
    if(X.size < 1): continue

    num_hits.append(X.size)
    global_X = np.append(global_X, X)
    global_Y = np.append(global_Y, Y)
    eta = -np.log(np.tan(np.arctan2(np.sqrt(X*X+Y*Y),Z)/2))
    centered_eta = eta - eta.mean(axis=0)
    global_eta = np.append(global_eta, eta)
    global_centered_eta = np.append(global_centered_eta, centered_eta)

    #phi = np.arctan2(Y,X)
    ### SO EPIC, how do I make phi right? Use imaginary numbers!
    imagY = Y*(0+1j)
    XY_vector = (X+imagY)/(np.abs(X+imagY))
    average_XY_vector = XY_vector.mean(axis=0)
    phi = np.angle(XY_vector)
    average_phi = np.angle(average_XY_vector)
    centered_phi = vdelta_phi(phi, average_phi)
    global_phi = np.append(global_phi, phi)
    global_centered_phi = np.append(global_centered_phi, centered_phi)

### Now we just plot!
plt.figure(1)
n, bins, patches = plt.hist(x=num_hits,
                            bins=[x-0.5 for x in range(20)],
                            color='#0000ff',
                            alpha=1.0,
                            rwidth=1.0)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('Number of hits')
plt.ylabel('Frequency')
plt.title('Number of hits per track')
plt.savefig('numberOfHits.png', bbox_inches='tight')

### 
plt.figure(2)
n, bins, patches = plt.hist(x=global_eta,
                            bins=eta_bins,
                            color='#ff0000',
                            alpha=1.0,
                            rwidth=1.0)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('$\eta$')
plt.ylabel('Frequency')
plt.title('$\eta$ of hits in the tracker')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.yscale('log', nonposy='clip')
plt.ylim(bottom=0.5,top=np.power(10,np.ceil(np.log10(maxfreq))))
plt.savefig('hitsEta_all.png', bbox_inches='tight')

### 
plt.figure(3)
n, bins, patches = plt.hist(x=global_phi,
                            bins=phi_bins,
                            color='#ff0000',
                            alpha=1.0,
                            rwidth=1.0)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('$\phi$')
plt.ylabel('Frequency')
plt.title('$\phi$ of hits in the tracker')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.yscale('log', nonposy='clip')
plt.ylim(bottom=0.5,top=np.power(10,np.ceil(np.log10(maxfreq))))
plt.savefig('hitsPhi_all.png', bbox_inches='tight')

### 
plt.figure(4)
n, bins, patches = plt.hist(x=global_centered_eta,
                            bins=delta_bins,
                            color='#ff0000',
                            alpha=1.0,
                            rwidth=1.0)
mean = np.average(global_centered_eta)
std = np.std(global_centered_eta)
print("Eta = ",mean,"+-",std)

plt.grid(axis='y', alpha=0.75)
plt.xlabel('$\Delta\eta$')
plt.ylabel('Frequency')
plt.title('Dispersion of track hits around their mean ($\eta$)')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.yscale('log', nonposy='clip')
plt.ylim(bottom=0.5,top=np.power(10,np.ceil(np.log10(maxfreq))))
plt.savefig('hitsDeltaEta_all.png', bbox_inches='tight')


plt.figure(5)
n, bins, patches = plt.hist(x=global_centered_phi,
                            bins=delta_bins,
                            color='#ff0000',
                            alpha=1.0,
                            rwidth=1.0)
plt.grid(axis='y', alpha=0.75)
plt.xlabel('$\Delta\phi$')
plt.ylabel('Frequency')
plt.title('Dispersion of track hits around their mean ($\phi$)')
maxfreq = n.max()
# Set a clean upper y-axis limit.
plt.yscale('log', nonposy='clip')
plt.ylim(bottom=0.5,top=np.power(10,np.ceil(np.log10(maxfreq))))
plt.savefig('hitsDeltaPhi_all.png', bbox_inches='tight')
mean = np.average(global_centered_phi)
std = np.std(global_centered_phi)
print("Phi = ",mean,"+-",std)

plt.figure(6)
colors = (0,0,0)
area = 1
plt.scatter(global_X, global_Y, s=area, marker=",")
plt.title('(X,Y) position of hits')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('hitsXY_all.png', bbox_inches='tight')

plt.figure(7)
colors = (0,0,0)
area = 1
plt.hexbin(global_eta, global_phi, C=None, gridsize=20, cmap=cm.jet, bins=None)
plt.axis([global_eta.min(), global_eta.max(), global_phi.min(), global_phi.max()])
cb = plt.colorbar()
cb.set_label('Number of hits')
plt.savefig('hitsEtaPhi_hexagon_all.png', bbox_inches='tight')
print("Total number of hits is "+str(global_eta.size))

plt.figure(8)
heatmap, xedges, yedges = np.histogram2d(global_eta, global_phi,
                                         bins=([x/10 for x in range(-44,45,4)],
                                               make_phi_range(13)))
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
print(xedges)
print(yedges)

plt.imshow(heatmap.T, extent=extent, origin='lower', cmap=cm.afmhot, vmin=-0, vmax=500)
cb = plt.colorbar()
cb.set_label('Number of hits')
plt.savefig('hitsEtaPhi_square_all.png', bbox_inches='tight')
plt.xlabel('$\eta$')
plt.ylabel('$\phi$')

sys.exit(0)

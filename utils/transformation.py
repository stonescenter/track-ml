import numpy as np

def rho_from_xy(x, y):
    return np.sqrt(x*x+y*y)

def eta_from_theta(theta):
    return -np.log(np.tan(theta/2.0))

def eta_from_xyz(x, y, z):
    theta = np.arctan2(np.sqrt(x*x + y*y),z)
    return eta_from_theta(theta)

def phi_from_xy(x, y):
    return np.arctan2(y,x)

def convert_xyz_to_rhoetaphi(x, y, z):
    return rho_from_xy(x,y), eta_from_xyz(x,y,z), phi_from_xy(x,y)

def convert_rhoetaphi_to_xyz(rho, eta, phi):
    x = rho*np.cos(phi)
    y = rho*np.sin(phi)
    z = rho*np.sinh(eta)
    return x, y, z

def translate_hit_deta_dphi(hit, deta, dphi):
    # Input: hit with X, Y, Z
    rho, eta, phi = convert_xyz_to_rhoetaphi(hit[0],hit[1],hit[2])
    eta = eta+deta
    phi = phi+dphi
    while (phi > np.pi):   phi -= 2*np.pi
    while (phi <= -np.pi): phi += 2*np.pi
    x, y, z = convert_rhoetaphi_to_xyz(rho,eta,phi)
    return np.array([x,y,z])

def rotate_hit(hit):
    y = hit[1]
    x = hit[0]
    theta = np.arctan2(y,x)
    theta = -theta
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s,0), (s, c,0), (0,0,1)))
    rotated_hit = R.dot(hit)
    return rotated_hit

def rotate_hit_by_angle(theta, hit):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s,0), (s, c,0), (0,0,1)))
    rotated_hit = R.dot(hit)
    return rotated_hit

if __name__ == "__main__":
    print("Test 1: rotate hit to horizontal")
    hit = np.array([np.random.random(), np.random.random(), np.random.random()])
    r_hit = rotate_hit(hit)
    print(hit)
    print(r_hit)

    print("\nTest 2: test pt, eta, phi <-> x, y, z conversions")
    for i in range(1000000):
        if(i%50000 == 0):
            print(i)
        v1 = np.random.randint(low=-1000, high=1000)
        v2 = np.random.randint(low=-1000, high=1000)
        v3 = np.random.randint(low=-1000, high=1000)
        if(v1==0 and v2==0):
            continue
        v = np.array([v1*np.random.random(),
                      v2*np.random.random(),
                      v3*np.random.random()])
        rho, eta, phi = convert_xyz_to_rhoetaphi(v[0], v[1], v[2])
        vback = np.array(convert_rhoetaphi_to_xyz(rho, eta, phi))
        np.testing.assert_allclose(vback,v,1e-07,1e-12)

import numpy as np


def rho_from_xy(x, y):
    return np.sqrt(x * x + y * y)


def eta_from_theta(theta):
    return -np.log(np.tan(theta / 2.0))


def eta_from_xyz(x, y, z):
    theta = np.arctan2(np.sqrt(x * x + y * y), z)
    return eta_from_theta(theta)


def phi_from_xy(x, y):
    return np.arctan2(y, x)


def convert_xyz_to_rhoetaphi(x, y, z):
    return rho_from_xy(x, y), eta_from_xyz(x, y, z), phi_from_xy(x, y)


def convert_rhoetaphi_to_xyz(rho, eta, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    z = rho * np.sinh(eta)
    return x, y, z


def translate_hit_deta_dphi(hit, deta, dphi):
    # Input: hit with X, Y, Z
    rho, eta, phi = convert_xyz_to_rhoetaphi(hit[0], hit[1], hit[2])
    eta = eta + deta
    phi = phi + dphi
    while phi > np.pi:
        phi -= 2 * np.pi
    while phi <= -np.pi:
        phi += 2 * np.pi
    x, y, z = convert_rhoetaphi_to_xyz(rho, eta, phi)
    return np.array([x, y, z])


def rotate_hit(hit):
    y = hit[1]
    x = hit[0]
    theta = np.arctan2(y, x)
    theta = -theta
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    rotated_hit = R.dot(hit)
    return rotated_hit


def rotate_hit_by_angle(theta, hit):
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    rotated_hit = R.dot(hit)
    return rotated_hit


def make_hits_rhoetaphi_collection(hits):
    global_X = np.empty(0)
    global_Y = np.empty(0)
    global_rho = np.empty(0)
    global_eta = np.empty(0)
    global_phi = np.empty(0)
    global_hits = np.empty(0)
    global_tech1 = np.empty(0)
    global_tech2 = np.empty(0)
    global_tech3 = np.empty(0)

    # Loop over particles
    for i in range(hits.shape[0]):
        the_hits = hits[i]
        reshaped_hits = the_hits.reshape(int(114 / 6), 6)
        X = reshaped_hits[:, 0]
        Y = reshaped_hits[:, 1]
        Z = reshaped_hits[:, 2]
        tech1 = reshaped_hits[:, 3]
        tech2 = reshaped_hits[:, 4]
        tech3 = reshaped_hits[:, 5]
        X = np.trim_zeros(X)
        Y = np.trim_zeros(Y)
        Z = np.trim_zeros(Z)
        tech1 = tech1[: X.size]
        tech2 = tech2[: X.size]
        tech3 = tech3[: X.size]

        assert (
            X.size == Y.size
            and X.size == Z.size
            and Y.size == Z.size
            and tech1.size == X.size
            and tech2.size == X.size
            and tech3.size == X.size
        )

        if X.size < 1:
            continue
        #  conversion for eta, phi
        rho = rho_from_xy(X, Y)
        eta = eta_from_xyz(X, Y, Z)
        phi = phi_from_xy(X, Y)
        global_rho = np.append(global_rho, rho)
        global_eta = np.append(global_eta, eta)
        global_phi = np.append(global_phi, phi)
        global_tech1 = np.append(global_tech1, tech1)
        global_tech2 = np.append(global_tech2, tech2)
        global_tech3 = np.append(global_tech3, tech3)

    global_hits = np.column_stack(
        (global_rho, global_eta, global_phi, global_tech1, global_tech2, global_tech3)
    )
    return global_hits


def select_hits_eta_phi(global_hits, eta_min, eta_max, phi_min, phi_max):

    cell_hits = global_hits[
        np.where(
            (global_hits[:, 1] > eta_min)
            * (global_hits[:, 1] < eta_max)
            * (global_hits[:, 2] > phi_min)
            * (global_hits[:, 2] < phi_max)
        )
    ]
    return cell_hits


def parts_from_tracks(tracks):
    # Slice the input file in its constituent parts
    indexes = tracks[:, 0]  # Particle indexes (0,1,2,...)
    vertices = tracks[:, 1:4]  # Particle vertex (tx,ty,tz)
    momenta = tracks[:, 4:7]  # Particle momentum (px,py,pz)
    hits = tracks[
        :, 7:-1
    ]  # N hits with the following information: 3D point + 3 tech. info
    return indexes, vertices, momenta, hits


if __name__ == "__main__":
    print("Test 1: rotate hit to horizontal")
    hit = np.array([np.random.random(), np.random.random(), np.random.random()])
    r_hit = rotate_hit(hit)
    print(hit)
    print(r_hit)

    print("\nTest 2: test pt, eta, phi <-> x, y, z conversions")
    for i in range(1000000):
        if i % 50000 == 0:
            print(i)
        v1 = np.random.randint(low=-1000, high=1000)
        v2 = np.random.randint(low=-1000, high=1000)
        v3 = np.random.randint(low=-1000, high=1000)
        if v1 == 0 and v2 == 0:
            continue
        v = np.array(
            [v1 * np.random.random(), v2 * np.random.random(), v3 * np.random.random()]
        )
        rho, eta, phi = convert_xyz_to_rhoetaphi(v[0], v[1], v[2])
        vback = np.array(convert_rhoetaphi_to_xyz(rho, eta, phi))
        np.testing.assert_allclose(vback, v, 1e-07, 1e-12)

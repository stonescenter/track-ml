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

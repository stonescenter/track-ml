import numpy as np

def rotate_hit(hit):
    y = hit[1]
    x = hit[0]
    theta = np.arctan2(y,x)
    theta = -theta
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s,0), (s, c,0), (0,0,1)))
    rotated_hit = R.dot(hit)
    return rotated_hit

if __name__ == "__main__":
    hit = np.array([np.random.random(), np.random.random(), np.random.random()])
    r_hit = rotate_hit(hit)
    print(hit)
    print(r_hit)

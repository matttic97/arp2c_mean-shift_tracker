from ex2_utils import *
import matplotlib.pyplot as plt
from ms_algorithm import *
from time import time


def generate_responses_2():
    responses = np.zeros((200, 200), dtype=np.float32)
    responses[70, 80] = 0.5
    responses[40, 100] = 0.4
    responses[97, 71] = 0.3
    responses[120, 60] = 1
    responses[100, 50] = 0.4
    return gausssmooth(responses, 10)


def run_test(start_loc, scales=3, n_iter=30):
    for scale in range(scales):
        threshold = 0.025
        h = 5+scale*2
        Xi, Yi = create_ms_kernel(h, h)
        gKernel = create_gauss_kernel(h, h)
        image = generate_responses_2()

        t = time()
        x_k1 = (1, 1)
        iterations = 0
        while abs(x_k1[0]) > threshold and abs(x_k1[1]) > threshold: # and iterations < n_iter:
            patch, mask = get_patch(image, start_loc, (h, h))
            x_k1 = get_mean_shift_vector(patch, Xi, Yi, gKernel)
            start_loc += x_k1
            iterations += 1

        print('kernel size:', h)
        print('duration:', time()-t, 'seconds,', iterations, 'iterations')
        print('last diff:', x_k1)
        print('max found at:', np.floor(start_loc))
        print('----')


# run test
start_loc = np.array([75.0,88.0])
run_test(start_loc, scales=4)
plt.imshow(generate_responses_2())
plt.plot([75.0], [88.0], 'x', color='r')
plt.savefig('ms-mode.jpg')
plt.show()

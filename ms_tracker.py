from ex2_utils import *
from ms_algorithm import *

class MeanShiftTracker(Tracker):

    def initialize(self, image, region):

        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]

        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.size = (region[2]+abs(region[2]%2-1), region[3]+abs(region[3]%2-1))
        self.template, _ = get_patch(image, self.position, self.size)
        self.nbins = 8
        self.kernel_g = create_epanechnik_kernel(self.template.shape[1], self.template.shape[0], 10)
        self.kernel_h = create_uniform_kernel(self.kernel_g.shape[1], self.kernel_g.shape[0])
        self.Xi, self.Yi = create_ms_kernel(self.kernel_g.shape[1], self.kernel_g.shape[0])
        self.q = extract_histogram(self.template, self.nbins, self.kernel_g)

    def track(self, image):
        error = 1
        x_k1 = (1, 1)
        iterations = 0
        while abs(x_k1[0]) > self.parameters.threshold and abs(x_k1[1]) > self.parameters.threshold and iterations < 30:
            patch, mask = get_patch(image, self.position, self.size)
            p = extract_histogram(patch, self.nbins, self.kernel_g)
            v = np.sqrt(self.q / (p + 0.00001))
            w = backproject_histogram(patch, v, self.nbins)
            x_k1 = get_mean_shift_vector(w, self.Xi, self.Yi, self.kernel_h)
            error = np.linalg.norm(x_k1, 2)
            self.position = (self.position[0] + x_k1[0], self.position[1] + x_k1[1])
            iterations += 1

        if error < 0.052:
            self.q = (1-self.parameters.alpha)*self.q + self.parameters.alpha*p

        return [self.position[0] - self.size[0]/2, self.position[1]-self.size[1]/2, self.size[0], self.size[1]]


class MSParams():
    def __init__(self):
        self.enlarge_factor = 2
        self.threshold = 0.01
        self.alpha = 0.01
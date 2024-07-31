import numpy as np
from typing import List, Any
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


np.random.seed(200)

x_size = 500
A_probability = 0.5
q = 1
z_mean = 0
sparsity_arr = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75]
A_size_arr = [100, 150, 200, 250, 300, 350, 400]
sigma_arr = [1]


def plot_mse(mse: List[Any]) -> None:
    fig, ax0 = plt.subplots(nrows=1)
    im = ax0.pcolormesh(mse)
    fig.colorbar(im, ax=ax0)
    plt.savefig("initial_sims.png")


def simulate() -> List[Any]:
    mse_arr = []

    for i in range(len(A_size_arr)):
        mse_arr.append([])
        for j in range(len(sparsity_arr)):
            y_size = A_size_arr[i]
            A = np.random.binomial(1, A_probability, (A_size_arr[i], 500))
            sparsity = sparsity_arr[j]
            x_true = np.zeros(x_size)
            num_non_zero = int(sparsity * x_size)
            non_zero_indices = np.random.choice(x_size, num_non_zero, replace=False)
            x_true[non_zero_indices] = np.random.uniform(1, 1000, num_non_zero)
            z_stddev = sigma_arr[0]
            z_size = y_size
            z = np.random.normal(z_mean, z_stddev, z_size)
            y = [np.dot(A[i], x_true)*(1+q)**(z[i]) for i in range(y_size)]
            y = np.array(y)
            y_tilda = y/(1+q)**(z_stddev**2/2)

            lasso = Lasso()
            lasso.fit(A, y_tilda)
            x = lasso.coef_

            mse = np.sqrt(mean_squared_error(x, x_true))/np.linalg.norm(x_true)
            mse_arr[i].append(mse)
    return mse_arr


if __name__ == "__main__":
    print("Hi")
    mse = simulate()
    plot_mse(mse)

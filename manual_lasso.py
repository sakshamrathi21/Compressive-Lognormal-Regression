import numpy as np
from typing import List, Any
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from typing import List, Any


np.random.seed(200)

x_size = 500
A_probability = 0.5
q = 1
z_mean = 0
sparsity_arr = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.75]
A_size_arr = [100, 150, 200, 250, 300, 350, 400]
sigma_arr = [4]
lambda_values = np.logspace(-4, 0, 50)


def plot_mse(mse: List[Any], file_name) -> None:
    fig, ax = plt.subplots()
    cax = ax.matshow(mse, cmap='viridis')
    fig.colorbar(cax)
    
    ax.set_xticks(np.arange(len(sparsity_arr)))
    ax.set_xticklabels([str(s) for s in sparsity_arr])
    ax.set_xlabel('Sparsity')

    ax.set_yticks(np.arange(len(A_size_arr)))
    ax.set_yticklabels([str(a) for a in A_size_arr])
    ax.set_ylabel('A Size')
    
    plt.title('MSE vs Sparsity and A Size')
    plt.savefig(file_name)


def simulate(sigma) -> List[Any]:
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
            z_stddev = sigma
            z_size = y_size
            z = np.random.normal(z_mean, z_stddev, z_size)
            y = [np.dot(A[i], x_true)*(1+q)**(z[i]) for i in range(y_size)]
            y = np.array(y)
            y_tilda = y/(1+q)**(z_stddev**2/2)
            A_train, A_test, y_train, y_test = train_test_split(A, y_tilda, test_size=0.2, random_state=42)
            best_lambda = None
            best_mse = float('inf')
            for lambda_val in lambda_values:
                lasso = Lasso(alpha=lambda_val)
                lasso.fit(A_train, y_train)
                y_pred = lasso.predict(A_test)
                mse = mean_squared_error(y_test, y_pred)
                
                if mse < best_mse:
                    best_mse = mse
                    best_lambda = lambda_val
            print(f"The best lambda value for the size of A = {A_size_arr[i]} and sparsity of {sparsity_arr[j]} is: {best_lambda}.")
            lasso = Lasso(alpha=best_lambda)
            lasso.fit(A, y_tilda)
            x = lasso.coef_
            mse = np.sqrt(mean_squared_error(x, x_true))/np.linalg.norm(x_true)
            mse_arr[i].append(np.log(mse))
    return mse_arr


def simulate_without_corrections(sigma) -> List[Any]:
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
            z_stddev = sigma
            z_size = y_size
            z = np.random.normal(z_mean, z_stddev, z_size)
            y = [np.dot(A[i], x_true)*(1+q)**(z[i]) for i in range(y_size)]
            y = np.array(y)
            y_tilda = y
            A_train, A_test, y_train, y_test = train_test_split(A, y_tilda, test_size=0.2, random_state=42)
            best_lambda = None
            best_mse = float('inf')
            for lambda_val in lambda_values:
                lasso = Lasso(alpha=lambda_val)
                lasso.fit(A_train, y_train)
                y_pred = lasso.predict(A_test)
                mse = mean_squared_error(y_test, y_pred)
                
                if mse < best_mse:
                    best_mse = mse
                    best_lambda = lambda_val
            print(f"The best lambda value for the size of A = {A_size_arr[i]} and sparsity of {sparsity_arr[j]} is: {best_lambda}.")
            lasso = Lasso(alpha=best_lambda)
            lasso.fit(A, y_tilda)
            x = lasso.coef_

            mse = np.sqrt(mean_squared_error(x, x_true))/np.linalg.norm(x_true)
            mse_arr[i].append(np.log(mse))
    return mse_arr


if __name__ == "__main__":
    print("Hi")
    for sigma in sigma_arr:
        mse = simulate(sigma)
        plot_mse(mse, f"initial_sims_with_correction_{sigma}.png")
        # print(mse)
        mse = simulate_without_corrections(sigma)
        plot_mse(mse, f"initial_sims_without_correction_{sigma}.png")


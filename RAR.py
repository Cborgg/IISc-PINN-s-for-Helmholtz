import numpy as np
import deepxde as dde
from deepxde.backend import tf
import matplotlib.pyplot as plt

def helmholtz_pinn(k, num_domain=10000, num_test=1000):
    geom = dde.geometry.Interval(0.0, 1.0)

    def pde(x, y):
        d2p = dde.grad.hessian(y, x, i=0)
        return d2p + k**2 * y

    bc0 = dde.DirichletBC(
        geom, lambda x: 1.0,
        lambda x, on_b: on_b and np.isclose(x[0], 0.0),
    )
    bc1 = dde.DirichletBC(
        geom, lambda x: -1.0,
        lambda x, on_b: on_b and np.isclose(x[0], 1.0),
    )

    def true_solution(x):
        A = 1.0
        B = (-1.0 - A * np.cos(k * 1.0)) / np.sin(k * 1.0)
        return A * np.cos(k * x[:, 0:1]) + B * np.sin(k * x[:, 0:1])

    data = dde.data.PDE(
        geom,
        pde,
        [bc0, bc1],
        num_domain=num_domain,
        num_boundary=2,
        num_test=num_test,
        solution=true_solution,
    )

    net = dde.maps.FNN([1, 64, 64, 64, 1], "tanh", "Glorot uniform")

    def output_transform(x, y):
        x_ = x[:, 0:1]
        phi1 = x_
        phi2 = 1 - x_
        return phi2 * 1.0 + phi1 * (-1.0) + phi1 * phi2 * y

    net.apply_output_transform(output_transform)

    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, metrics=["l2 relative error"])
    model.train(epochs=20000)
    model.compile("L-BFGS", metrics=["l2 relative error"])
    model.train()

    # Plotting
    x_test = np.linspace(0, 1, num_test)[:, None]
    p_pred = model.predict(x_test)[:, 0]
    p_true = true_solution(x_test)[:, 0]
    rel_err = np.abs(p_pred - p_true) / np.maximum(1e-6, np.abs(p_true))

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(x_test, p_true, "k-", label="Exact")
    plt.plot(x_test, p_pred, "r--", label="PINN")
    plt.xlabel("x"); plt.ylabel("p(x)")
    plt.legend(); plt.title(f"Helmholtz (k={k:.2f})")

    plt.subplot(1, 2, 2)
    plt.plot(x_test, rel_err, "b-")
    plt.xlabel("x"); plt.ylabel("Relative error")
    plt.title("Pointwise relative error")

    plt.tight_layout(); plt.show()

if __name__ == "__main__":
    c = 340.0      # speed of sound [m/s]
    f = 1529.0     # frequency [Hz]
    k = 2 * np.pi * f / c
    helmholtz_pinn(k)


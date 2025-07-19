import deepxde as dde
import numpy as np
from deepxde.backend import tf
import skopt
from distutils.version import LooseVersion
import matplotlib.pyplot as plt



def quasirandom(n_samples, sampler):
    """
    Generate quasirandom samples in [0,1] for 1D.
    """
    space = [(0.0, 1.0)]
    if sampler == "LHS":
        sampler_obj = skopt.sampler.Lhs(
            lhs_type="centered", criterion="maximin", iterations=1000
        )
    elif sampler == "Halton":
        sampler_obj = skopt.sampler.Halton(min_skip=-1, max_skip=-1)
    elif sampler == "Hammersley":
        sampler_obj = skopt.sampler.Hammersly(min_skip=-1, max_skip=-1)
    elif sampler == "Sobol":
        if LooseVersion(skopt.__version__) < LooseVersion("0.9"):
            sampler_obj = skopt.sampler.Sobol(min_skip=2, max_skip=2, randomize=False)
        else:
            sampler_obj = skopt.sampler.Sobol(skip=0, randomize=False)
        pts = np.array(sampler_obj.generate(space, n_samples + 2))[2:]
        return pts.reshape(-1, 1)
    else:
        raise ValueError(f"Unknown sampler '{sampler}'")
    pts = np.array(sampler_obj.generate(space, n_samples))
    return pts.reshape(-1, 1)


def main(k, NumDomain, method):
    """
    Solve 1D Helmholtz PINN with various sampling strategies.
    p'' + k^2 p = 0,  p(0)=1, p(1)=-1
    """
    # PDE residual
    def pde(x, y):
        d2p = dde.grad.hessian(y, x, i=0)
        return d2p + k**2 * y

    # Analytical solution
    def true_solution(x):
        A = 1.0
        B = (-1.0 - A * np.cos(k * 1.0)) / np.sin(k * 1.0)
        return A * np.cos(k * x[:, 0:1]) + B * np.sin(k * x[:, 0:1])

    # Geometry and BCs
    geom = dde.geometry.Interval(0.0, 1.0)
    bc0 = dde.DirichletBC(geom, lambda x: 1.0,
                          lambda x, on_b: on_b and np.isclose(x[0], 0.0))
    bc1 = dde.DirichletBC(geom, lambda x: -1.0,
                          lambda x, on_b: on_b and np.isclose(x[0], 1.0))

    # Build data object
    if method in ['Grid', 'Random']:
        data = dde.data.PDE(
            geom,
            pde,
            [bc0, bc1],
            num_domain=NumDomain,
            num_boundary=2,
            num_test=1000,
            train_distribution='uniform' if method == 'Grid' else 'pseudo',
            solution=true_solution,
        )
    elif method in ['LHS', 'Halton', 'Hammersley', 'Sobol']:
        anchors = quasirandom(NumDomain, method)
        data = dde.data.PDE(
            geom,
            pde,
            [bc0, bc1],
            num_domain=0,
            num_boundary=2,
            num_test=1000,
            train_distribution='uniform',
            solution=true_solution,
            anchors=anchors,
        )
    else:
        raise ValueError(f"Unknown method '{method}'")

    # Neural network
    net = dde.maps.FNN([1, 64, 64, 64, 1], "tanh", "Glorot uniform")

    # Enforce BCs via output transform
    def output_transform(x, y):
        x0 = x[:, 0:1]
        phi1 = x0
        phi2 = 1 - x0
        return phi2 * 1.0 + phi1 * (-1.0) + phi1 * phi2 * y

    net.apply_output_transform(output_transform)

    # Model
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3, metrics=["l2 relative error"]);
    losshistory, train_state = model.train(epochs=5000)
    x_test = np.linspace(0, 1, 500)[:, None]
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
    # Report error
    error = np.array(losshistory.metrics_test)[-1]
    print("L2 relative error:", error)
    return error


if __name__ == "__main__":
    c = 343.0
    f = 1000.0
    k = 2 * np.pi * f / c
    main(k=k, NumDomain=2000, method='Grid')

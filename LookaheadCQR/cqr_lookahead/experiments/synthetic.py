import matplotlib

matplotlib.use('TkAgg')

from sklearn.model_selection import train_test_split
from LookaheadCQR.cqr_lookahead.uncertainty import CQR
from LookaheadCQR.lookahead.models.lookahead import Lookahead
from LookaheadCQR.lookahead.models.models import polyRegression
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import torch
import numpy as np
import LookaheadCQR.lookahead.models.uncertainty as uncert
import LookaheadCQR.lookahead.models.prediction as pred
import LookaheadCQR.lookahead.models.propensity as prop

np.set_printoptions(precision=3)


# Helper functions to print performance
def get_perf(model, xs, ys, eta, mask, x, uncert=False):
    perf = {'mse': [], 'mae': [], 'improve': [], 'imprate': [], 'contain': [], 'size': []}
    perf['mse'].append([model.mse(x_, y_) for x_, y_ in zip(xs, ys)])
    perf['mae'].append([model.mae(x_, y_) for x_, y_ in zip(xs, ys)])
    perf['improve'].append([model.improve(x_, y_, eta, mask) for x_, y_ in zip(xs, ys)])
    perf['imprate'].append(
        [model.improve_rate(x_, y_, eta, mask) for x_, y_ in zip(xs, ys)])
    if uncert:
        xsp = [model.move_points(x_) for x_ in xs]
        perf['contain'].append([model.contain(x_)[0] for x_ in [*xsp, x]])
        perf['size'].append([model.contain(x_)[1] for x_ in [*xsp, x]])
    perf = {k: np.asarray(v) for k, v in zip(perf.keys(), perf.values())}
    return perf


def print_perf(perf, idx=0, uncert=False):
    print('\ttrn\ttst\tall')
    print(('mse' + '\t{:.4f}' * 3).format(*perf['mse'][idx, :]))
    print(('mae' + '\t{:.4f}' * 3).format(*perf['mae'][idx, :]))
    print(('imprv' + '\t{:.4f}' * 3).format(*perf['improve'][idx, :]))
    print(('imprt' + '\t{:.4f}' * 3).format(*perf['imprate'][idx, :]))
    print()
    if uncert:
        print('\ttrn\'\ttst\'\tall')
        print(('contn' + '\t{:.3f}' * 3).format(*perf['contain'][idx, :]))
        print(('intrsz' + '\t{:.3f}' * 3).format(*perf['size'][idx, :]))
        print()


# Helper function for plotting
def plot_quad(ax, M, x, y, x_plot, y_plot, eta, mask, color, lw=1, osz=50, olw=1,
              fill_between_color = 'g'):
    x_plot_th = torch.from_numpy(x_plot.astype(np.float32))
    ax.plot(x_plot, y_plot, '--', color="tab:orange", linewidth =1.0, label ="gt", alpha=0.6)

    f_pred = M.f.predict(x_plot)
    ax.plot(x_plot, f_pred, label="f", linewidth=lw, alpha=0.6)

    if M.u is not None:
        u_pred, l_pred = M.u.lu(x_plot_th)
        u_pred = u_pred.detach().numpy()
        l_pred = l_pred.detach().numpy()
        ax.fill_between(x_plot.flatten(), u_pred.flatten(), l_pred.flatten(),
                        color=fill_between_color, alpha=0.05, zorder=0)
        ax.plot(x_plot.flatten(), u_pred.flatten(), f"tab:{color}", linewidth=1.0, alpha=0.5, zorder=0)
        ax.plot(x_plot.flatten(), l_pred.flatten(), f"tab:{color}", linewidth=1.0, alpha=0.5, zorder=0)

    xp = M.move_points(x, eta, mask)
    ypstar = M.fstar.predict(xp)
    ax.scatter(x, y, color="white", edgecolor="tab:blue", alpha = 1, s=osz, zorder=10, linewidth=olw)
    ax.scatter(xp, ypstar, color="white", edgecolor="tab:red", alpha = 1, s=osz, zorder=10, linewidth=olw)

    ax.set_xticks([])
    ax.set_yticks([])


## Defining the ground truth function f*
class Fstar_parabola():
    def __init__(self, coeffs=[]):
        self.coeffs = coeffs

    def fit(self, x, y):
        pass

    def predict(self, x):
        c = self.coeffs[-1]
        for i in range(2, len(self.coeffs) + 1):
            c = self.coeffs[-i] + c * x
        return c.flatten()

class Fstar_sin():
    def __init__(self, coeffs=[]):
        self.coeffs = coeffs

    def fit(self, x, y):
        pass

    def predict(self, x):
        c = self.coeffs[-1]
        for i in range(2, len(self.coeffs) + 1):
            c = self.coeffs[-i] + c * np.sin(x / (i - 1))
        return c.flatten()

class Fstar_exp():
    def __init__(self, coeffs=[]):
        self.coeffs = coeffs

    def fit(self, x, y):
        pass

    def predict(self, x):
        c = self.coeffs[-1]
        for i in range(2, len(self.coeffs) + 1):
            c = self.coeffs[-i] + c * np.exp(x / (i - 1))
        return c.flatten()


def synthetic_exp(eta, fstar, name):
    print("\n\n")
    print(f"{'*' * 25}")
    print(f"EXPERIMENT: {name}")
    print(f"{'*' * 25}")
    print("\n\n")

    ## Setting the data hyperparameters
    # Random Seed
    seed = 3
    np.random.seed(seed)

    # Number of Training Samples
    n = 1000

    # Our input is drawn from the normal distribution
    # with std = sig and mean = offset
    sig = 0.5
    offset = -0.8

    # std of noise added to the data
    ns = 0.1

    # Split ratio for train-test
    trn_sz = 0.75

    # Degree of polynomial regressor
    degree = 2

    x = np.random.normal(size=(n, 1), scale=sig) + offset
    y = fstar.predict(x)
    y += np.random.normal(scale=ns, size=y.shape)

    n, d = x.shape
    x_trn, x_tst, y_trn, y_tst = train_test_split(x, y, test_size=1 - trn_sz,
                                                  random_state=seed)
    n_trn, n_tst = (x_trn.shape[0], x_tst.shape[0])
    print("n:", n, ", n_trn:", n_trn, ", n_tst:", n_tst)
    xs = [x_trn, x_tst, x]
    ys = [y_trn, y_tst, y]

    ## Setting the model hyperparamters
    # l1/l2 Regularization coefficient
    alpha = 0.

    # Lam controls the tradeoff between accuracy and decision improvement
    lam = 4.

    " Hyperparamteres for Lookahead"
    # Number of cycles for training
    num_cycles = 10
    # Z-score controls size of confidence intervas
    z_score = 1.65  # for confiednce intervals (1.28-90%, 1.65=95%)

    """ Hyperparameters for Prediction Model"""
    # Learning rate
    lr_f = 0.05

    # Number of training iterations
    num_iter_init = 1000
    num_iter_f = 100
    num_iter_base = num_iter_init + num_iter_f * num_cycles

    """ Hyperparameters for Uncertanity Model"""
    # number of bootstrapped models
    num_gs = 10

    # Learning rate
    lr_g = 0.001

    # Number of training iterations
    num_iter_g = 5000  # for training g in cycles

    """ Mask"""
    # mask[i] is 1 if it can be changed for making decisions
    mask = np.ones(d)
    print('mask:', mask)

    ## Training the baseline model with no lookahead regularization
    # train baseline
    verbose = True

    print('training baseline:')
    f_model = polyRegression(1, 1, degree)
    f_base = pred.PredModel(d, model=f_model, reg_type='none', alpha=alpha, lr=lr_f,
                            num_iter_init=num_iter_base)
    model_base = Lookahead(f_base, None, None, lam=0., eta=eta, mask=mask,
                           ground_truth_model=fstar)
    _, _ = model_base.train(x_trn, y_trn, num_cycles=0, random_state=seed,
                            verbose=verbose)

    perf_base = get_perf(model_base, xs, ys, eta, mask, x)
    print_perf(perf_base)

    ## Training the paper's lookahead model
    # train our model
    verbose = True

    print('Training lookahead:')
    f_model = polyRegression(1, 1, degree)
    g_model = polyRegression(1, 1, degree)

    f = pred.PredModel(d, model=f_model, reg_type='none', alpha=0., lr=lr_f,
                       num_iter=num_iter_f, num_iter_init=num_iter_init)
    u = uncert.Bootstrap(d, model=g_model, alpha=0., num_gs=num_gs, z_score=z_score,
                         lr=lr_g, num_iter=num_iter_g)
    h = prop.PropModel(random_state=seed)
    model = Lookahead(f, u, h, lam=lam, eta=eta, mask=mask, ground_truth_model=fstar)
    _, _ = model.train(x_trn, y_trn, num_cycles=num_cycles, random_state=seed,
                       verbose=verbose)

    ## Training our CQR + lookahead model
    # train our model
    verbose = True

    print('Training lookahead:')
    f_model = polyRegression(1, 1, degree)

    f = pred.PredModel(d, model=f_model, reg_type='none', alpha=0., lr=lr_f,
                       num_iter=num_iter_f, num_iter_init=num_iter_init)
    u = CQR(d, tau=(0.05, 0.95), lr=lr_g, num_iter=num_iter_g)
    h = prop.PropModel(random_state=seed)
    nn_cqr_model = Lookahead(f, u, h, lam=lam, eta=eta, mask=mask,
                             ground_truth_model=fstar)
    _, _ = nn_cqr_model.train(x_trn, y_trn, num_cycles=num_cycles, random_state=seed,
                              verbose=verbose)

    ## Comparing the performance of all models
    perf_base = get_perf(model_base, xs, ys, eta, mask, x)
    perf_la = get_perf(model, xs, ys, eta, mask, x, uncert=True)
    perf_la_cqr = get_perf(nn_cqr_model, xs, ys, eta, mask, x, uncert=True)

    print('\nBaseline Model:')
    print_perf(perf_base)
    print('Lookahead Model:')
    print_perf(perf_la, uncert=True)
    print('Lookahead+CQR Model:')
    print_perf(perf_la_cqr, uncert=True)

    ## Plotting the results
    plt.rcParams['figure.figsize'] = (6.0, 4.5)
    x_plot = np.linspace(-3.0, 3.0, 1000)[:, np.newaxis]
    y_plot = fstar.predict(x_plot)

    fig, ax = plt.subplots()
    axins = inset_axes(ax, width="40%", height="40%", loc=4)
    plot_quad(axins, model_base, x, y, x_plot, y_plot, eta, mask, osz=10, olw=0.5,
              color='green')
    axins.set_xlim([-2, 3.3])
    axins.set_ylim([-7, 1.0])

    plot_quad(ax, model, x, y, x_plot, y_plot, eta, mask, lw=2, color='green')
    ax.set_xlim([-2.2, 2.2])
    ax.set_ylim([-4, 1])
    #plt.title(r'$\eta={}$'.format(eta))
    plt.title(r'${}$'.format(name))

    plot_quad(ax, nn_cqr_model, x, y, x_plot, y_plot, eta, mask, lw=2, color='purple',
              fill_between_color='m')
    ax.set_xlim([-2.2, 2.2])
    ax.set_ylim([-4, 1])
    plt.title(r'${}$'.format(name))
    #plt.title(r'$\eta={}$'.format(eta))


    ax.legend(
        [
            r'$f^*$',
            r'$f$',
            r'$Lookahead$',
            r'$Lookahead + CQR$',
            r'$(x,y)$',
            r"$(x',y')$"
        ],
        fontsize=8,
    )


# Coefficients of the ground truth polynomial and sinusodial
# coeffs[i] denotes the coefficient of x^i th term
coeffs = [0.1, 0.5, -0.8]

fstar_poly = Fstar_parabola(coeffs)
fstar_sin = Fstar_sin(coeffs)
fstar_exp = Fstar_exp([0.8,0.3,-0.2])
fstars = (fstar_poly, fstar_sin,fstar_exp)
f_name = ['Poly','Sinus','Exp']

# Decision step size
etas = (0.4, 0.8, 1.25, 2)
"""etas = [0.4]
fstars = [fstar_exp]
f_name = ['Exp']"""
# Experiments names
if __name__ == '__main__':
    for i, f_star in enumerate(fstars):
        for eta_ in etas:
            name_ = f" eta={eta_}, {f_name[i]}"
            synthetic_exp(eta_, f_star, name_)
    plt.show()

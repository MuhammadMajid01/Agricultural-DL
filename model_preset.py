
import matplotlib.pyplot as plt
import numpy as np


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))
fig.suptitle('LOSS', fontsize=14, fontweight='bold')


loss_param = np.arange(1, 21)


train_loss_a = 10 + (90 * np.exp(-loss_param/3))
val_loss_a = 10 + (90 * np.exp(-loss_param/4))


train_loss_b = 10 + (70 * np.exp(-loss_param/3))
val_loss_b = 10 + (70 * np.exp(-loss_param/4))


train_loss_c = 5 + (45 * np.exp(-loss_param/3))
val_loss_c = 5 + (45 * np.exp(-loss_param/4))


ax1.plot(loss_param, train_loss_a, label='train', marker='o', linewidth=2)
ax1.plot(loss_param, val_loss_a, label='val', marker='s', linewidth=2)
ax1.set_title('(a)')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.set_ylim(0, 105) 
ax1.legend()
ax1.grid(True)



ax2.plot(loss_param, train_loss_b, label='train', marker='o', linewidth=2)
ax2.plot(loss_param, val_loss_b, label='val', marker='s', linewidth=2)
ax2.set_title('(b)')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Loss')
ax2.set_ylim(0, 85)
ax2.legend()
ax2.grid(True)




ax3.plot(loss_param, train_loss_c, label='train', marker='o', linewidth=2)
ax3.plot(loss_param, val_loss_c, label='val', marker='s', linewidth=2)
ax3.set_title('(c)')
ax3.set_xlabel('Epochs')
ax3.set_ylabel('Loss')
ax3.set_ylim(0, 55)  
ax3.legend()
ax3.grid(True)



plt.subplots_adjust(hspace=0.5) 


plt.show()



# 

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


np.random.seed(42)


n_points = 100
true_sizes = np.random.uniform(0.1, 10, n_points)

pred_sizes_a = true_sizes * np.random.normal(1.0, 0.02, n_points)  
pred_sizes_b = true_sizes * np.random.uniform(0.8, 1.2, n_points)  


fig, axs = plt.subplots(2, 2, figsize=(12, 12))


axs[0, 0].scatter(true_sizes, pred_sizes_a, alpha=0.6, edgecolors='w')
axs[0, 0].plot([0, 10], [0, 10], 'r--', linewidth=1)
axs[0, 0].set_title('(a) Predicted vs Ground Truth (High Accuracy)', pad=20)
axs[0, 0].set_xlabel('Ground truth size')
axs[0, 0].set_ylabel('Predicted sizes')
axs[0, 0].set_xlim(0, 10)
axs[0, 0].set_ylim(0, 10)
axs[0, 0].grid(True, linestyle='--', alpha=0.5)


axs[0, 1].scatter(true_sizes, pred_sizes_b, alpha=0.6, edgecolors='w')
axs[0, 1].plot([0, 10], [0, 10], 'r--', linewidth=1)
axs[0, 1].set_title('(b) Predicted vs Ground Truth (Lower Accuracy)', pad=20)
axs[0, 1].set_xlabel('Ground truth sizes')
axs[0, 1].set_ylabel('Predicted sizes')
axs[0, 1].set_xlim(0, 10)
axs[0, 1].set_ylim(0, 10)
axs[0, 1].grid(True, linestyle='--', alpha=0.5)

errors_a = pred_sizes_a - true_sizes
kde_a = gaussian_kde(errors_a)
x_vals = np.linspace(-1, 1, 200)
axs[1, 0].plot(x_vals, kde_a(x_vals), 'b-', linewidth=2)
axs[1, 0].fill_between(x_vals, kde_a(x_vals), color='b', alpha=0.2)
axs[1, 0].axvline(0, color='r', linestyle='--', linewidth=1)
axs[1, 0].set_title('(c) Error Distribution (High Accuracy)', pad=20)
axs[1, 0].set_xlabel('Prediction error (Predicted - Ground truth)')
axs[1, 0].set_ylabel('Density')
axs[1, 0].grid(True, linestyle='--', alpha=0.5)

errors_b = pred_sizes_b - true_sizes
kde_b = gaussian_kde(errors_b)
x_vals = np.linspace(-5, 5, 200)
axs[1, 1].plot(x_vals, kde_b(x_vals), 'b-', linewidth=2)
axs[1, 1].fill_between(x_vals, kde_b(x_vals), color='b', alpha=0.2)
axs[1, 1].axvline(0, color='r', linestyle='--', linewidth=1)
axs[1, 1].set_title('(d) Error Distribution (Lower Accuracy)', pad=20)
axs[1, 1].set_xlabel('Prediction error (Predicted - Ground truth)')
axs[1, 1].set_ylabel('Density')
axs[1, 1].grid(True, linestyle='--', alpha=0.5)


plt.subplots_adjust(top=0.92, bottom=0.08, hspace=0.4, wspace=0.3)
plt.show()

# 

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(42)


FIG_WIDTH = 2
FIG_HEIGHT = 6


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(FIG_WIDTH, FIG_HEIGHT))
plt.subplots_adjust(hspace=0.6)


num_points = 150
confidence_level = 0.98
base_scale = 0.05

x = np.linspace(0, 1, num_points)
y1 = np.random.normal(0, base_scale * 0.8, num_points)
y2 = np.random.normal(0, base_scale * 1.2, num_points)
y3 = np.random.normal(0, base_scale * 0.6, num_points) 
y4 = np.random.normal(0, base_scale * 1.5, num_points) 


z_value = norm.ppf(1 - (1 - confidence_level)/2)
theoretical_bound = base_scale * z_value


def plot_errors(ax, y_data, colors):
    for y, color in zip(y_data, colors):
        ax.scatter(x, y, alpha=0.7, s=40, color=color)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax.axhline(y=theoretical_bound, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax.axhline(y=-theoretical_bound, color='green', linestyle='--', linewidth=1, alpha=0.7)
    ax.fill_between(x, -theoretical_bound, theoretical_bound, color='green', alpha=0.1)
    ax.set_ylim(-0.3, 0.3)
    ax.set_xlim(0, 1)
    ax.set_ylabel('Actual Prediction Error')
    ax.set_xlabel('Predicted Uncertainty')


plot_errors(ax1, [y1, y2], ['blue', 'orange'])
ax1.set_title('Prediction Error Analysis', pad=15)


plot_errors(ax2, [y3, y4], ['green', 'red'])
ax2.set_title('Prediction Error Analysis', pad=15)
plot_errors(ax3, [y1, y2, y3, y4], ['blue', 'orange', 'green', 'red'])
ax3.set_title('Combined Prediction Errors', pad=15)
plt.tight_layout()
plt.show()
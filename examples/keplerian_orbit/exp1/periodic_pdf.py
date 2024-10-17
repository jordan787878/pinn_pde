import numpy as np
import matplotlib.pyplot as plt


def joint_normal_pdf(x1, x2, mu1, sigma1, mu2, sigma2):
    """Compute the joint normal PDF."""
    return (1 / (sigma1 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x1 - mu1) / sigma1) ** 2) * \
           (1 / (sigma2 * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x2 - mu2) / sigma2) ** 2)


def periodic_joint_pdf(x1, x2, mu1, sigma1, mu2, sigma2, num_terms=10):
    """Compute the periodic joint PDF."""
    pdf_periodic = np.zeros((len(x1), len(x2)))
    # Sum the shifted normal distributions for x2
    for n in range(-num_terms, num_terms + 1):
        shifted_x2 = x2 + 2 * np.pi * n
        pdf_periodic += joint_normal_pdf(np.meshgrid(x1, indexing='ij')[0], shifted_x2[:, None], mu1, sigma1, mu2, sigma2)
    return pdf_periodic


# Parameters
mu1 = 8000       # Mean for x1
sigma1 = 100    # Std for x1
mu2 = 0       # Mean for x2
sigma2 = np.pi/10    # Std for x2
x1 = np.linspace(7000, 9000, 100)     # x1 range
x2 = np.linspace(0.0, 2*np.pi, 100)  # x2 range

# Compute periodic joint PDF
pdf = periodic_joint_pdf(x1, x2, mu1, sigma1, mu2, sigma2)

# Normalize the PDF over the domain
pdf_normalized = pdf / np.trapz(np.trapz(pdf, x=x1), x=x2)

# Plotting
X1, X2 = np.meshgrid(x1, x2)
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X1, X2, pdf_normalized, cmap='viridis', edgecolor='none')
ax.set_title('Periodic Joint Normal Distribution (Surface Plot)')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('Density')
plt.show()
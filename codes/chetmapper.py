import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_cylinder(ax, N1, N2, L, R, phi_offset_L1, phi_offset_L2, bundles_green=None, label=None, colorin = "blue", colorout = "red"):
    """
    Disegna un cilindro stereo 3D su un asse esistente (ax)
    - N1, N2: numero fibre layer 1 e 2
    - L: lunghezza cilindro
    - R: raggio
    - phi_offset_L1, phi_offset_L2: offset di calibrazione dei layer
    - bundles_green: lista di indici da evidenziare in verde
    - label: etichetta per legenda
    """
    if bundles_green is None:
        bundles_green = []

    z = np.linspace(-L, L, 400)

    def plot_fiber(phi0, direction, color, lw):
        phi = (phi0 + direction * ((z + L) / (2*L)) * np.pi) % (2*np.pi)
        x = R * np.cos(phi)
        y = R * np.sin(phi)
        ax.plot(x, y, z, lw=lw, color=color)

    # Layer 1
    for i in range(N1):
        phi0 = (-2*np.pi * i / N1 + phi_offset_L1) % (2*np.pi)
        lw = 2.5 if i in bundles_green else 0.25
        color = colorin if i in bundles_green else colorin
        plot_fiber(phi0, direction=-1, color=color, lw=lw)

    # Layer 2
    for i in range(N2):
        phi0 = (2*np.pi * i / N2 + phi_offset_L2) % (2*np.pi)
        global_idx = i + N1
        lw = 2.5 if global_idx in bundles_green else 0.25
        color = colorout if global_idx in bundles_green else colorout
        plot_fiber(phi0, direction=+1, color=color, lw=lw)

    if label:
        ax.text(0, 0, L + 2, label, fontsize=12, color='k')



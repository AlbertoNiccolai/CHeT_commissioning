import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def mapper_plot_c1(bundles_green, N1=45, N2=49, L=15., R=1.7):

    # ----------------------------------------
    #  Calibrazione
    # ----------------------------------------
    DELTA_ROT = np.deg2rad(10)
    PHI_OFFSET_L1 = 4.118974897606805   # layer 1
    PHI_OFFSET_L2 = 3.654505739890168    # layer 2
    PHI_OFFSET_L1 += DELTA_ROT
    PHI_OFFSET_L2 += DELTA_ROT

    # ----------------------------------------
    #  Setup figura 3D
    # ----------------------------------------
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111, projection='3d')
    z = np.linspace(-L, L, 400)

    # ----------------------------------------
    #  Funzione generica per disegnare una fibra
    # ----------------------------------------
    def plot_fiber(phi0, direction, color, lw):
        """
        phi0     = angolo di partenza a z = -L
        direction = +1 (layer 2) oppure -1 (layer 1)
        """
        phi = (phi0 + direction * ((z + L) / (2*L)) * np.pi) % (2*np.pi)
        x = R * np.cos(phi)
        y = R * np.sin(phi)
        ax.plot(x, y, z, lw=lw, color=color)

    # ----------------------------------------
    #  Disegna tutto il layer 1 (nero)
    # ----------------------------------------
    for i in range(N1):
        phi0 = (-2*np.pi * i / N1 + PHI_OFFSET_L1) % (2*np.pi)
        plot_fiber(phi0, direction=-1, color="blue", lw=0.2)

    # ----------------------------------------
    #  Disegna tutto il layer 2 (nero)
    # ----------------------------------------
    for i in range(N2):
        phi0 = (2*np.pi * i / N2 + PHI_OFFSET_L2) % (2*np.pi)
        plot_fiber(phi0, direction=+1, color="red", lw=0.2)

    # ----------------------------------------
    #  Evidenzia i bundles selezionati (verde)
    # ----------------------------------------
    for b in bundles_green:

        if b < N1:  
            # Layer 1
            i = b
            phi0 = (-2*np.pi * i / N1 + PHI_OFFSET_L1) % (2*np.pi)
            plot_fiber(phi0, direction=-1, color="blue", lw=2.0)

        else:
            # Layer 2
            i = b - N1
            phi0 = (2*np.pi * i / N2 + PHI_OFFSET_L2) % (2*np.pi)
            plot_fiber(phi0, direction=+1, color="red", lw=2.0)

        #print(phi0)

    # ----------------------------------------
    #  Aspetto della figura
    # ----------------------------------------
    ax.set_title(f"CHet C1 3D plot - Bundles fired: {bundles_green}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-R-1, R+1)
    ax.set_ylim(-R-1, R+1)
    ax.set_zlim(-L, L)
    ax.view_init(elev=7, azim=11)

    # Etichette US / DS
    ax.text(0, 0, -L-2, "US", color='k', fontsize=16)
    ax.text(0, 0, +L+1, "DS - Readout", color='k', fontsize=16)

    plt.tight_layout()
    plt.show()


# ----------------------
#  Esempio
# ----------------------
mapper_plot_c1([30, 62])

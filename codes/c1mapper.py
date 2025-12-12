def mapper_plot(bundles_green, N1=45, N2=49, L=15.):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111, projection='3d')

    R = 1.7

    # ====================================================
    # 1) Funzioni helper
    # ====================================================
    def phi_layer1(i):
        # phi0 uniforme su [0,2π)
        return (2*np.pi * i / N1)

    def phi_layer2(i):
        return (2*np.pi * i / N2)

    # inclinazioni (versi definiti)
    def phi_z(phi0, z, sign):
        return phi0 + sign * ((z + L) / (2*L)) * np.pi


    # ====================================================
    # 2) Disegno layer 1 (blu) - verso +φ (sinistra)
    # ====================================================
    for i in range(N1):
        phi0 = phi_layer1(i)
        z = np.linspace(-L, L, 200)
        phi = phi_z(phi0, z, sign=+1)
        ax.plot(R*np.cos(phi), R*np.sin(phi), z, lw=0.2, color='blue')


    # ====================================================
    # 3) Disegno layer 2 (rosso) - verso -φ (destra)
    # ====================================================
    for i in range(N2):
        phi0 = phi_layer2(i)
        z = np.linspace(-L, L, 200)
        phi = phi_z(phi0, z, sign=-1)
        ax.plot(R*np.cos(phi), R*np.sin(phi), z, lw=0.2, color='red')


    # ====================================================
    # 4) Bundles verdi (liste)
    # ====================================================
    for b in bundles_green:

        if b < N1:
            # Layer 1
            i = b
            phi0_b = phi_layer1(i)
            sign = +1
            color = "blue"
        else:
            # Layer 2
            i = b - N1
            phi0_b = phi_layer2(i)
            sign = -1
            color = "red"

        z = np.linspace(-L, L, 200)
        phi_b = phi_z(phi0_b, z, sign=sign)

        ax.plot(R*np.cos(phi_b), R*np.sin(phi_b), z, lw=2.5, color=color)


    # ====================================================
    # 5) Setup grafico
    # ====================================================
    ax.set_title(f"CHet C1 3D plot - Bundles fired: {bundles_green}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-R-1, R+1)
    ax.set_ylim(-R-1, R+1)
    ax.set_zlim(-L, L)

    ax.view_init(elev=7, azim=11)
    ax.text(0,0,-20, "US", fontsize=18)
    ax.text(0,0, 20, "DS - Readout", fontsize=18)

    plt.tight_layout()
    plt.show()


# Test
mapper_plot([37, 54])

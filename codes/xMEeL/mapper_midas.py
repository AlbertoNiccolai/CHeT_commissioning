#def mapper_plot(bundles_green, N1=45, N2=49, L=15.):
#    import numpy as np
#    import matplotlib.pyplot as plt
#    from mpl_toolkits.mplot3d import Axes3D
#
#    fig = plt.figure(figsize=(16,9))
#    ax = fig.add_subplot(111, projection='3d')
#
#
#    phi_starts_1 = []
#    R = 3
#    for i in range (45):
#        phi_starts_1.append(2*np.pi * i / 45.)
#    
#    phi_starts_2 = []
#    for i in range (49):
#        phi_starts_2.append(2*np.pi * i / 49.)
#
#    # plot di ciascun bundle
#    for phi0 in phi_starts_1:
#        z = np.linspace(-15, 15, 200)
#        phi = phi0 + ( (z + 15) / 30 ) * np.pi   # mezzo giro
#        x = R * np.cos(phi)
#        y = R * np.sin(phi)
#
#        ax.plot(x, y, z, lw=0.1, color='black')
#    
#    # plot di ciascun bundle
#    for phi0 in phi_starts_2:
#        z = np.linspace(-15, 15, 200)
#        phi = phi0 - ( (z + 15) / 30 ) * np.pi   # mezzo giro
#        x = R * np.cos(phi)
#        y = R * np.sin(phi)
#
#        ax.plot(x, y, z, lw=0.1, color='black')
#    
#    z = np.linspace(-15, 15, 200)
#    bund_1 = 27
#    phi0_bund_1 = 2*np.pi * bund_1 / 45.
#    phi_bund_1 = -6.911503837897545 + phi0_bund_1 + ( (z + 15) / 30 ) * np.pi   # mezzo giro
#    print(phi0_bund_1 + ( (15 + 15) / 30 ) * np.pi)   # mezzo giro
#    x = R * np.cos(phi_bund_1)
#    y = R * np.sin(phi_bund_1)
#    ax.plot(x, y, z, lw=1.6, color='green')
#
#    bund_2 = 63 - N1
#    phi0_bund_2 = 2*np.pi * bund_2 / 45.
#    phi_bund_2 = 0.6283185307179586 + phi0_bund_2 - ( (z + 15) / 30 ) * np.pi   # mezzo giro
#    print(phi0_bund_2 - ( (15 + 15) / 30 ) * np.pi)
#    x2 = R * np.cos(phi_bund_2)
#    y2 = R * np.sin(phi_bund_2)
#    ax.plot(x2, y2, z, lw=1.6, color='green')
#    
#    
#    ax.set_xlabel("x")
#    ax.set_ylabel("y")
#    ax.set_zlabel("z")
#    ax.set_xlim(-15,15)
#    ax.set_ylim(-15,15)
#    plt.tight_layout()
#    plt.show()





def mapper_plot(bundles_green, N1=45, N2=49, L=15.):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111, projection='3d')

    R = 1.7

    # =========================
    # 1) Bundles neri
    # =========================
    # Gruppo 1 (45)
    for i in range(N1):
        phi0 = 2*np.pi * (-i) / N1 +5.1661745859032155  + 1.46608
        z = np.linspace(-L, L, 200)
        phi = phi0 - ((z + L) / (2*L)) * np.pi
        ax.plot(R*np.cos(phi), R*np.sin(phi), z, lw=0.2, color='blue')
        #print("LAYER 1:")
        #print(f"phi-15 = {phi0}")
        #print(f"phi+15 = {phi0 + ((15 + L) / (2*L)) * np.pi}")
        
    
    

    # Gruppo 2 (49)
    for i in range(N2):
        phi0 = 2*np.pi * (i) / N2 -1.1540544441758425 + 1.13097
        if phi0 < 0:
            phi0 = phi0 + 2*np.pi
        z = np.linspace(-L, L, 200)
        phi = phi0 + ((z + L) / (2*L)) * np.pi
        ax.plot(R*np.cos(phi), R*np.sin(phi), z, lw=0.2, color='red')
        #print("LAYER 2:")
        #print(f"phi-15 = {phi0}")
        #print(f"phi+15 = {phi0 + ((15 + L) / (2*L)) * np.pi}")

    # =========================
    # 2) Bundles verdi (lista)
    # =========================
    for b in bundles_green:

        if b < N1:
            # Gruppo 1 → rotazione positiva
            phi0_b = 2*np.pi * (-b) / N1 +5.1661745859032155  + 1.46608 - np.pi
            direction = -1
            color = "blue"
        else:
            # Gruppo 2 → rotazione negativa
            b2 = b - N1
            phi0_b = 2*np.pi * (b2) / N2  -1.1540544441758425 + 1.13097 - np.pi
            if phi0_b < 0:
                phi0_b = phi0_b + 2*np.pi
            direction = +1
            color = "red"

        z = np.linspace(-L, L, 200)
        phi_b = phi0_b + direction * ((z + L) / (2*L)) * np.pi
        print(phi0_b + direction * ((-15 + L) / (2*L)) * np.pi)

        ax.plot(R*np.cos(phi_b), R*np.sin(phi_b), z, lw=2, color=color)

    # =========================
    # 3) Assi e visualizzazione
    # =========================
    ax.set_title(f"CHet C1 3D plot - Bundles fired: {bundles_green} - z = 15 -> US")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-R-1, R+1)
    ax.set_ylim(-R-1, R+1)
    ax.set_zlim(-L, L)
    # Imposta vista di default
    ax.view_init(elev=7, azim=11)
    ax.text(x=0, y=0, z=-20, s="US", color='k', fontsize=18)
    ax.text(x=0, y=0, z=20, s="DS - Readout", color='k', fontsize=18)

    plt.tight_layout()
    plt.show()


mapper_plot([37,54])


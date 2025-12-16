import numpy as np
import re
import matplotlib.pyplot as plt


from channelsToBundles import channel_bundle_map_board0_C2, channel_bundle_map_board1_C2, get_bundle_0, get_bundle_1
def get_bundle_0_2(channel):
    return channel_bundle_map_board0_C2.get(channel, None)

def get_bundle_1_2(channel):
    return channel_bundle_map_board1_C2.get(channel, None)


file_list = [
    ("/Users/Alberto/Downloads/Run101_list.txt", 0),
    ("/Users/Alberto/Downloads/Run104_list.txt", 360 -  15),
    ("/Users/Alberto/Downloads/Run105_list.txt", 360 -  30),
    ("/Users/Alberto/Downloads/Run107_list.txt", 360 - 45),
    ("/Users/Alberto/Downloads/Run108_list.txt", 360 - 60),
    ("/Users/Alberto/Downloads/Run109_list.txt", 360 - 75),
    ("/Users/Alberto/Downloads/Run110_list.txt", 360 - 90),
    ("/Users/Alberto/Downloads/Run111_list.txt", 360 - 105),
    ("/Users/Alberto/Downloads/Run112_list.txt", 360 - 120),
    ("/Users/Alberto/Downloads/Run113_list.txt", 360 - 135),
    ("/Users/Alberto/Downloads/Run114_list.txt", 360 - 150),
    ("/Users/Alberto/Downloads/Run115_list.txt", 360 - 165),
    ("/Users/Alberto/Downloads/Run117_list.txt", 360 - 180),
    ("/Users/Alberto/Downloads/Run118_list.txt", 360 - 345),
    ("/Users/Alberto/Downloads/Run119_list.txt", 360 - 330),
    ("/Users/Alberto/Downloads/Run120_list.txt", 360 - 315),
    ("/Users/Alberto/Downloads/Run121_list.txt", 360 - 300),
    ("/Users/Alberto/Downloads/Run122_list.txt", 360 - 285),
    ("/Users/Alberto/Downloads/Run123_list.txt", 360 - 270),
    ("/Users/Alberto/Downloads/Run124_list.txt", 360 - 255),
    ("/Users/Alberto/Downloads/Run125_list.txt", 360 - 240),
    ("/Users/Alberto/Downloads/Run127_list.txt", 360 - 225),
    ("/Users/Alberto/Downloads/Run128_list.txt", 360 - 210),
    ("/Users/Alberto/Downloads/Run129_list.txt", 360 - 195),

]

file_list_1 =[
    ("/Users/Alberto/Downloads/Run55_list.txt", 0),
    ("/Users/Alberto/Downloads/Run7_list.txt",  345),
    ("/Users/Alberto/Downloads/Run9_list.txt",  330),
    ("/Users/Alberto/Downloads/Run13_list.txt", 315),
    ("/Users/Alberto/Downloads/Run15_list.txt", 300),
    ("/Users/Alberto/Downloads/Run17_list.txt", 285),
    ("/Users/Alberto/Downloads/Run19_list.txt", 360 - 90),
    ("/Users/Alberto/Downloads/Run21_list.txt", 360 - 105),
    ("/Users/Alberto/Downloads/Run23_list.txt", 360 - 120),
    ("/Users/Alberto/Downloads/Run24_list.txt", 360 - 135),
    ("/Users/Alberto/Downloads/Run27_list.txt", 360 - 150),
    ("/Users/Alberto/Downloads/Run29_list.txt", 360 - 165),
    ("/Users/Alberto/Downloads/Run31_list.txt", 360 - 180),
    ("/Users/Alberto/Downloads/Run33_list.txt", 360 - 345),
    ("/Users/Alberto/Downloads/Run35_list.txt", 360 - 330),
    ("/Users/Alberto/Downloads/Run37_list.txt", 360 - 315),
    ("/Users/Alberto/Downloads/Run39_list.txt", 360 - 300),
    ("/Users/Alberto/Downloads/Run41_list.txt", 360 - 285),
    ("/Users/Alberto/Downloads/Run43_list.txt", 360 - 270),
    ("/Users/Alberto/Downloads/Run45_list.txt", 360 - 255),
    ("/Users/Alberto/Downloads/Run47_list.txt", 360 - 240),
    ("/Users/Alberto/Downloads/Run49_list.txt", 360 - 225),
    ("/Users/Alberto/Downloads/Run51_list.txt", 360 - 210),
    ("/Users/Alberto/Downloads/Run53_list.txt", 360 - 195),
]

noise_filename_1 = "/Users/Alberto/Downloads/Run12_list.txt"
noise_filename = "/Users/Alberto/Downloads/Run116_list.txt"


# -------------------------------------------------------------
# Funzioni di base (dalla tua geometria dei bundle)
# -------------------------------------------------------------
def bundle_to_fiber_params(b, N1, N2, PHI_OFFSET_L1, PHI_OFFSET_L2):
    if b < N1:
        i = b
        phi0 = (-2*np.pi * i / N1 + PHI_OFFSET_L1) % (2*np.pi)
        direction = -1
    else:
        i = b - N1
        phi0 = (2*np.pi * i / N2 + PHI_OFFSET_L2) % (2*np.pi)
        direction = +1
    return phi0, direction

def bundle_intersection(b1, b2, N1, N2, PHI_OFFSET_L1, PHI_OFFSET_L2, L):
    phi01, d1 = bundle_to_fiber_params(b1, N1, N2, PHI_OFFSET_L1, PHI_OFFSET_L2)
    phi02, d2 = bundle_to_fiber_params(b2, N1, N2, PHI_OFFSET_L1, PHI_OFFSET_L2)
    if d1 == d2:
        return None
    m1 = d1 * np.pi / (2*L)
    m2 = d2 * np.pi / (2*L)
    for k in [-1,0,1]:
        phi02_k = phi02 + k*2*np.pi
        z_int = (phi02_k - phi01)/(m1 - m2) - L
        if -L <= z_int <= L:
            phi_int = (phi01 + m1*(z_int + L)) % (2*np.pi)
            return phi_int, z_int
    return None


# -------------------------------------------------------------
# Costruisco un dizionario con i rate relativi sorgente-bkg
# -------------------------------------------------------------
def build_rate_dict_subtracted(boards_on, boards_bkg):
    rate_dict = {}

    for brd in boards_on:
        bundles_on  = boards_on[brd]["bundles"]
        counts_on   = boards_on[brd]["counts"]

        bundles_bkg = boards_bkg[brd]["bundles"]
        counts_bkg  = boards_bkg[brd]["counts"]

        for b_on, c_on, c_bkg in zip(bundles_on, counts_on, counts_bkg):
            r_eff = c_on - c_bkg
            if r_eff > 0:
                rate_dict[b_on] = r_eff

    return rate_dict


# -------------------------------------------------------------
# Parser ASCII Janus (come prima)
# -------------------------------------------------------------
def parse_janus_ascii(path, get_bundle_0_2, get_bundle_1_2):
    boards = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 3:
                continue
            try:
                brd = int(parts[0])
                ch  = int(parts[1])
                cnt = int(parts[2])
            except ValueError:
                continue
            if brd not in boards:
                boards[brd] = {"bundles":[], "counts":[], "tstamp_us":None}
            if len(parts) >= 4:
                try:
                    tstamp = float(parts[3])
                    boards[brd]["tstamp_us"] = tstamp
                except ValueError:
                    pass
            if brd == 0:
                boards[brd]["bundles"].append(get_bundle_0_2(ch))
                boards[brd]["counts"].append(cnt)
            elif brd == 1:
                boards[brd]["bundles"].append(get_bundle_1_2(ch))
                boards[brd]["counts"].append(cnt)
    return boards

# -------------------------------------------------------------
# Funzione principale: media pesata con prior della sorgente
# -------------------------------------------------------------
def reconstruct_position_with_prior(rate_dict, phi_true, z_true,
                                    delta_L0, delta_L1,
                                    N1, N2, L, RATE_THRESHOLD=500):
    layer0_ids = [b for b in rate_dict if b < N1 and rate_dict[b] > RATE_THRESHOLD]
    layer1_ids = [b for b in rate_dict if b >= N1 and rate_dict[b] > RATE_THRESHOLD]

    if not layer0_ids or not layer1_ids:
        return None, None
    print("========================")    
    print(f"LAYER 0 = {layer0_ids}")
    print(f"LAYER 1 = {layer1_ids}")
    print("========================")
    
    phi_vals = []
    z_vals   = []
    weights  = []

    for b0 in layer0_ids:
        for b1 in layer1_ids:
            res = bundle_intersection(b0, b1, N1, N2, delta_L0, delta_L1, L)
            if res is None:
                continue
            phi_int, z_int = res
            # peso: rate * prior sulla distanza dalla sorgente
            dphi = np.angle(np.exp(1j*(phi_int - phi_true)))  # wrap modulo 2π
            dz   = z_int - z_true
            sigma_phi = 0.01  # parametro da regolare
            sigma_z   = 0.5  # parametro da regolare
            w_prior = np.exp(-(dphi**2)/(2*sigma_phi**2) - (dz**2)/(2*sigma_z**2))
            w_bundle = rate_dict[b0] * rate_dict[b1]
            #w_total = w_prior * w_bundle
            w_total = 1 * w_bundle #Scelgo di non usare la prior
            if w_total <= 0:
                continue
            phi_vals.append(phi_int)
            z_vals.append(z_int)
            weights.append(w_total)

    if not weights:
        return None, None

    z_mean = np.average(z_vals, weights=weights)
    phi_vals = np.array(phi_vals)
    weights  = np.array(weights)
    x = np.average(np.cos(phi_vals), weights=weights)
    y = np.average(np.sin(phi_vals), weights=weights)
    phi_mean = np.arctan2(y, x) % (2*np.pi)
    print(f"i phi stimati sono: {phi_vals}, con media: {phi_mean}")
    print(f"le z stimate sono: {z_vals}, con media: {z_mean}")
    return phi_mean, z_mean

# -------------------------------------------------------------
# Plot delle intersezioni
# -------------------------------------------------------------
def plot_intersections(phi_vals, z_vals, weights, phi_rec=None, z_rec=None, phi_true=None, z_true=None):
    phi_vals = np.array(phi_vals)
    z_vals   = np.array(z_vals)
    weights  = np.array(weights)

    plt.figure(figsize=(10,6))
    sc = plt.scatter(np.rad2deg(phi_vals), z_vals, c=weights, s=weights*50, cmap='viridis', alpha=0.7)
    plt.colorbar(sc, label='Peso dell\'intersezione')
    if phi_rec is not None and z_rec is not None:
        plt.scatter(np.rad2deg(phi_rec), z_rec, color='red', marker='x', s=100, label='φ_rec, z_rec')
    if phi_true is not None and z_true is not None:
        plt.scatter(np.rad2deg(phi_true), z_true, color='blue', marker='o', s=100, label='φ_true, z_true')
    plt.xlabel('φ [deg]')
    plt.ylabel('z [cm]')
    plt.title('Intersezioni dei bundle con pesi')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

# -------------------------------------------------------------
# Plot della funzione di likelihood
# -------------------------------------------------------------
def plot_likelihood_grid_from_intersections(
    phi_ints, z_ints, weights,
    phi_rec=None, z_rec=None,
    phi_true=None, z_true=None,
    L=15., steps=150,
    sigma_phi=0.1, sigma_z=1.0
):
    phi_grid = np.linspace(0, 2*np.pi, steps)
    z_grid   = np.linspace(-L, L, steps)

    likelihood = np.zeros((steps, steps))

    phi_ints = np.array(phi_ints)
    z_ints   = np.array(z_ints)
    weights  = np.array(weights)

    for i, phi_g in enumerate(phi_grid):
        for j, z_g in enumerate(z_grid):
            dphi = np.angle(np.exp(1j*(phi_ints - phi_g)))
            dz   = z_ints - z_g
            likelihood[j, i] = np.sum(
                weights * np.exp(
                    -(dphi**2)/(2*sigma_phi**2)
                    -(dz**2)/(2*sigma_z**2)
                )
            )

    plt.figure(figsize=(10,6))
    plt.imshow(
        likelihood,
        extent=[0, 360, -L, L],
        origin='lower',
        aspect='auto',
        cmap='inferno'
    )
    plt.colorbar(label='Likelihood')

    if phi_rec is not None:
        plt.scatter(np.rad2deg(phi_rec), z_rec, c='cyan', s=80, marker='x', label='Ricostruita')

    if phi_true is not None:
        plt.scatter(np.rad2deg(phi_true), z_true, c='lime', s=80, marker='o', label='Vera')

    plt.xlabel('φ [deg]')
    plt.ylabel('z')
    plt.title('Likelihood HeatMap')
    plt.legend()
    plt.show()


# -------------------------------------------------------------
# Funzione per scrivere l'output: bundle - phi - zeta
# -------------------------------------------------------------
def dump_bundle_points(
    filename_out,
    rate_dict,
    PHI_OFFSET_L1, PHI_OFFSET_L2,
    N1, N2, L,
    RATE_THRESHOLD=0
):
    """
    Scrive su file punti (phi, z) associati ai singoli bundle.
    Ogni intersezione genera DUE righe: una per ciascun bundle.
    """

    layer0_ids = [b for b in rate_dict if b < N1 and rate_dict[b] > RATE_THRESHOLD]
    layer1_ids = [b for b in rate_dict if b >= N1 and rate_dict[b] > RATE_THRESHOLD]

    with open(filename_out, "a") as fout:
        for b0 in layer0_ids:
            for b1 in layer1_ids:

                res = bundle_intersection(
                    b0, b1,
                    N1, N2,
                    PHI_OFFSET_L1, PHI_OFFSET_L2,
                    L
                )

                if res is None:
                    continue

                phi, z = res

                # riga per bundle L0
                fout.write(
                    f"{b0}\t{phi:.6f}\t{z:.3f}\n"
                )

                # riga per bundle L1
                fout.write(
                    f"{b1}\t{phi:.6f}\t{z:.3f}\n"
                )



# -------------------------------------------------------------
# Esempio di loop su tutte le run
# -------------------------------------------------------------
L = 15.0
N1 = 45
N2 = 49

DELTA_ROT = 0#-0.04
PHI_OFFSET_L1 = 4.118974897606805 + DELTA_ROT
PHI_OFFSET_L2 = 3.654505739890168 + DELTA_ROT

CORRECTION_OFFSET_L1 = 0#-0.14
PHI_OFFSET_L1 = PHI_OFFSET_L1 + CORRECTION_OFFSET_L1

RATE_THRESHOLD = 300

#Preparo il file di output
output_txt = "bundle_points_z=5_C1.txt"
with open(output_txt, "w") as f:
    f.write("# bundle_id   phi_rad   zeta_cm\n")

# Liste per raccogliere errori
phi_errors = []
z_errors = []
phi_true_list = []
# file_list_1 contiene le tuple (path, phi_scan)
for filename, phi_deg in file_list:
    phi_true = np.deg2rad(phi_deg)
    z_true   = 5.0  # coordinata reale della sorgente
    boards_on  = parse_janus_ascii(filename, get_bundle_0_2, get_bundle_1_2)
    boards_bkg = parse_janus_ascii(noise_filename, get_bundle_0_2, get_bundle_1_2)

    rate_dict = build_rate_dict_subtracted(boards_on, boards_bkg)

    
  #  rate_dict = {b:r for b,r in zip(boards1[0]["bundles"] + boards1[1]["bundles"],
  #                                  boards1[0]["counts"] + boards1[1]["counts"])}
    
    #dump_bundle_points(
    #    output_txt,
    #    rate_dict,
    #    PHI_OFFSET_L1,
    #    PHI_OFFSET_L2,
    #    N1, N2, L,
    #    RATE_THRESHOLD=RATE_THRESHOLD
    #)
    
    # --- stima posizione con prior ---
    phi_rec, z_rec = reconstruct_position_with_prior(rate_dict, phi_true, z_true,
                                                     PHI_OFFSET_L1, PHI_OFFSET_L2,
                                                     N1, N2, L, RATE_THRESHOLD)
    
    if phi_rec is None or z_rec is None:
        continue
    
    # --- raccolta errori per istogrammi ---
    dphi = np.rad2deg(np.angle(np.exp(1j*(phi_rec - phi_true))))
    dz   = z_rec - z_true
    phi_errors.append(dphi)
    z_errors.append(dz)
    
    print(f"{filename}: φ_rec={np.rad2deg(phi_rec):.2f} deg, φ_true={phi_deg:.2f} deg, z_rec={z_rec:.2f}, z_true={z_true:.2f}, Δφ={dphi:.2f} deg, Δz={dz:.2f}")

    # --- plot intersezioni dei bundle ---
    phi_vals = []
    z_vals = []
    weights  = []
    layer0_ids = [b for b in rate_dict if b < N1 and rate_dict[b] > RATE_THRESHOLD]
    layer1_ids = [b for b in rate_dict if b >= N1 and rate_dict[b] > RATE_THRESHOLD]

    #print(layer0_ids)
    #print(layer1_ids)
    
    for b0 in layer0_ids:
        for b1 in layer1_ids:
            res = bundle_intersection(b0, b1, N1, N2, PHI_OFFSET_L1, PHI_OFFSET_L2, L)
            if res is None:
                continue
            phi_i, z_i = res
            w = rate_dict[b0]*rate_dict[b1]
            phi_vals.append(phi_i)
            z_vals.append(z_i)
            weights.append(w)
    
    phi_true_list.append(phi_deg)
    
    #print(f"phi vals: {len(phi_vals)}")
    #print(f"z vals: {len(z_vals)}")
    #print(f"weights: {len(weights)}")

    #print(f"phi rec: {len(phi_rec)}")
    #print(f"z rec: {len(z_rec)}")

    #print(f"phi true: {len(phi_true)}")
    #print(f"z true: {len(z_true)}")

    
    #plot_intersections(phi_vals, z_vals, weights, phi_rec, z_rec, phi_true, z_true)

    # --- plot heatmap della likelihood ---
    #plot_likelihood_grid_from_intersections(phi_rec, z_rec, weights, phi_rec, z_rec, phi_true, z_true,L=L, steps=24)

# -----------------------
# Istogrammi degli errori
# -----------------------
phi_errors = np.array(phi_errors)
z_errors   = np.array(z_errors)
phi_true_list = np.array(phi_true_list)

phi_mean = np.mean(phi_errors)
phi_var  = np.var(phi_errors, ddof=1)      # varianza campionaria
phi_std  = np.sqrt(phi_var)                # deviazione standard

z_mean = np.mean(z_errors)
z_var  = np.var(z_errors, ddof=1)
z_std  = np.sqrt(z_var)

plt.figure(figsize=(14,10))

# =======================
# Istogramma Δφ
# =======================
ax1 = plt.subplot(2,2,1)
ax1.hist(phi_errors, bins=15, color='skyblue', edgecolor='black')
ax1.axvline(phi_mean, color='red', linestyle='--', linewidth=2)

ax1.text(
    0.95, 0.95,
    f"mean = {phi_mean:.2f} deg\nσ = {phi_std:.2f} deg",
    transform=ax1.transAxes,
    ha='right', va='top',
    color='red'
)

ax1.set_xlabel('Residuals Δφ [deg]')
ax1.set_ylabel('Counts')
ax1.set_title(f'Angular residuals - thr rate = {RATE_THRESHOLD/10:.1f} Hz')


# =======================
# Istogramma Δz
# =======================
ax2 = plt.subplot(2,2,2)
ax2.hist(z_errors, bins=15, color='salmon', edgecolor='black')
ax2.axvline(z_mean, color='red', linestyle='--', linewidth=2)

ax2.text(
    0.95, 0.95,
    f"mean = {z_mean:.2f} cm\nσ = {z_std:.2f} cm",
    transform=ax2.transAxes,
    ha='right', va='top',
    color='red'
)

ax2.set_xlabel('Residuals Δz [cm]')
ax2.set_ylabel('Counts')
ax2.set_title(f'Z residuals - thr rate = {RATE_THRESHOLD/10:.1f} Hz')


# =======================
# Δφ vs φ_true
# =======================
ax3 = plt.subplot(2,2,3, sharex=None)
ax3.scatter(phi_true_list, phi_errors, color='blue', alpha=0.7)
ax3.axhline(0, color='black', linestyle='--', linewidth=1)

ax3.set_xlabel('φ true [deg]')
ax3.set_ylabel('Δφ [deg]')
ax3.set_title('Angular residuals vs φ')
ax3.grid(alpha=0.3)


# =======================
# Δz vs φ_true
# =======================
ax4 = plt.subplot(2,2,4, sharex=ax3)
ax4.scatter(phi_true_list, z_errors, color='darkred', alpha=0.7)
ax4.axhline(0, color='black', linestyle='--', linewidth=1)

ax4.set_xlabel('φ true [deg]')
ax4.set_ylabel('Δz [cm]')
ax4.set_title('Z residuals vs φ')
ax4.grid(alpha=0.3)


plt.tight_layout()
plt.show()







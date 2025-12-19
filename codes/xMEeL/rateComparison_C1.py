import numpy as np
import matplotlib.pyplot as plt
import re

channel_bundle_map_board0 = {
    31: 0, 27: 1, 23: 2, 19: 3, 15: 4, 11: 5, 7: 6, 3: 7,
    29: 8, 25: 9, 21: 10, 17: 11, 13: 12, 9: 13, 5: 14, 1: 15,
    0: 16, 4: 17, 8: 18, 12: 19, 16: 20, 20: 21, 24: 22, 28: 23,
    2: 24, 6: 25, 10: 26, 14: 27, 18: 28, 22: 29, 26: 30, 30: 31,
    63: 32, 59: 33, 55: 34, 51: 35, 47: 36, 43: 37, 39: 38, 35: 39,
    61: 40, 57: 41, 53: 42, 49: 43, 45: 44, 41: 45, 37: 46, 33: 47,
    32: 48, 36: 49, 40: 50, 44: 51, 48: 52, 52: 53, 56: 54, 60: 55,
    34: 56, 38: 57, 42: 58, 46: 59, 50: 60, 54: 61, 58: 62, 62: 63
}

channel_bundle_map_board1 = {
    31: 64, 27: 65, 23: 66, 19: 67, 15: 68, 11: 69, 7: 70, 3: 71,
    29: 72, 25: 73, 21: 74, 17: 75, 13: 76, 9: 77, 5: 78, 1: 79,
    0: 80, 4: 81, 8: 82, 12: 83, 16: 84, 20: 85, 24: 86, 28: 87,
    2: 88, 6: 89, 10: 90, 14: 91, 18: 92, 22: 93, 26: 94, 30: 95,
    63: 96, 59: 97, 55: 98, 51: 99, 47: 100, 43: 101, 39: 102, 35: 103,
    61: 104, 57: 105, 53: 106, 49: 107, 45: 108, 41: 109, 37: 110, 33: 111,
    32: 112, 36: 113, 40: 114, 44: 115, 48: 116, 52: 117, 56: 118, 60: 119,
    34: 120, 38: 121, 42: 122, 46: 123, 50: 124, 54: 125, 58: 126, 62: 127
}

import numpy as np
import matplotlib.pyplot as plt
import re

# -------------------------------------------------------------
# MAPPINGS (tuoi, lasciati così come sono)
# -------------------------------------------------------------

def get_bundle(board, ch):
    if board == 0:
        return channel_bundle_map_board0.get(ch)
    elif board == 1:
        return channel_bundle_map_board1.get(ch)
    else:
        return None


# -------------------------------------------------------------
# PARSER ROBUSTO
# -------------------------------------------------------------
def parse_janus_ascii(path):
    """
    Legge file ASCII Janus.
    Restituisce:
    {
        board_id : {
            "channels": [ch1, ch2, ...],
            "bundles" : [b1, b2, ...],
            "counts"  : [...],
            "tstamp_us": float
        }
    }
    """
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
            except:
                continue

            if brd not in boards:
                boards[brd] = {
                    "channels": [],
                    "bundles": [],
                    "counts": [],
                    "tstamp_us": None
                }

            # timestamp
            if len(parts) >= 4:
                try:
                    boards[brd]["tstamp_us"] = float(parts[3])
                except:
                    pass

            bundle = get_bundle(brd, ch)

            if bundle is None:
                print(f"[WARN] Board {brd}: canale {ch} non presente nel mapping!")

            boards[brd]["channels"].append(ch)
            boards[brd]["bundles"].append(bundle)
            boards[brd]["counts"].append(cnt)

    return boards


# -------------------------------------------------------------
# FUNZIONE ROBUSTA DI PLOT
# -------------------------------------------------------------
def plot_board(board_id, ref, cmp):
    T1 = ref["tstamp_us"] / 1e6
    T2 = cmp["tstamp_us"] / 1e6

    bundles_ref = np.array(ref["bundles"])
    counts_ref  = np.array(ref["counts"])
    bundles_cmp = np.array(cmp["bundles"])
    counts_cmp  = np.array(cmp["counts"])

    rate_ref = counts_ref / T1
    rate_cmp = counts_cmp / T2

    all_bundles = sorted(set(b for b in bundles_ref if b is not None)
                       | set(b for b in bundles_cmp if b is not None))

    def build_vector(all_b, b_list, r_list):
        out = np.zeros(len(all_b))
        for b, r in zip(b_list, r_list):
            if b is None:
                continue
            idx = all_b.index(b)
            out[idx] = r
        return out

    R_ref = build_vector(all_bundles, bundles_ref, rate_ref)
    R_cmp = build_vector(all_bundles, bundles_cmp, rate_cmp)
    diff  = R_ref - R_cmp

    plt.figure(figsize=(12,5))
    plt.bar(all_bundles, R_ref)
    plt.title(f"Board {board_id} – Source Rate")
    plt.xlabel("Bundle")
    plt.ylabel("Rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.figure(figsize=(12,5))
    plt.bar(all_bundles, R_cmp)
    plt.title(f"Board {board_id} – Background Rate")
    plt.xlabel("Bundle")
    plt.ylabel("Rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.figure(figsize=(12,5))
    plt.bar(all_bundles, diff)
    plt.title(f"Board {board_id} – (source - background)")
    plt.xlabel("Bundle")
    plt.ylabel("Difference Rate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.show()



# -------------------------------------------------------------
# ESEMPIO D'USO
# -------------------------------------------------------------
file1 = "/Users/Alberto/Downloads/Run29_list.txt"
file2 = "/Users/Alberto/Downloads/Run28_list.txt"

boards1 = parse_janus_ascii(file1)
boards2 = parse_janus_ascii(file2)

for board_id in boards1:
    if board_id in boards2:
        print(f"Plotting board {board_id}...")
        plot_board(board_id, boards1[board_id], boards2[board_id])
    else:
        print(f"Board {board_id} non presente nel secondo file.")

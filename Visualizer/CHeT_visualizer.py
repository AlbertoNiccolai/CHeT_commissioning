import sys
import os
import midas.file_reader
import struct
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

# Import custom classes
try:
    from AcquisitionModes import Spectroscopy, Timing, Counting
    from AcquisitionModes import DTQ_SPECT, DTQ_TSPECT, DTQ_TIMING, DTQ_COUNT
except ImportError:
    print("ERROR: File 'AcquisitionModes.py' not found.")
    sys.exit(1)

# ==========================================
# 0. GLOBALS
# ==========================================
N1 = 45
N2 = 49
N3 = 59
N4 = 60
L_FIBER = 15.0

# Geometrical Offsets
PHI_OFFSET_C1IN  = 4.293507822806237
PHI_OFFSET_C1OUT = 3.829038665089601
PHI_OFFSET_C2IN  = 2.609122003
PHI_OFFSET_C2OUT = 3.351032122

TOA_MAX_GLOBAL = 40000  # Hard cutoff for Time of Arrival (Noise filter)

# ==========================================
# 1. READING FUNCTIONS
# ==========================================

def read_nth_physics_event(filename):
    """MODE 1: Reads a specific single physics event."""
    try:
        req_index = int(input("\n>>> Which physics event do you want to analyze? (0=first, 1=second...): "))
        req_n_hits = int(input(">>> How many hits to visualize? "))
    except ValueError:
        print("Invalid input.")
        return []
    if req_index < 0: return []
    
    print(f"\nScanning file {filename}...")
    try:
        mfile = midas.file_reader.MidasFile(filename)
    except Exception as e:
        print(f"Error: {e}"); return []

    physics_counter = -1
    target_event = None
    
    for event in mfile:
        if event.header.is_midas_internal_event(): continue
        if event.header.event_id != 1: continue
        physics_counter += 1
        if physics_counter == req_index:
            target_event = event
            break
    
    if target_event is None:
        print(f"❌ Event #{req_index} not found."); return []
    print(f"✅ Found! (Serial: {target_event.header.serial_number})")

    all_hits = [] 
    for bank_name, bank in target_event.banks.items():
        raw = bytes(bank.data)
        if len(raw) < 4: continue
        dtq = struct.unpack("<I", raw[:4])[0]
        if dtq == DTQ_TIMING:
            timing_ev = Timing(raw)
            if timing_ev.nhits > 0:
                try: match = re.search(r'\d+', bank_name); board_id = int(match.group()) if match else 0
                except: board_id = 0
                ref = getattr(timing_ev, 'fine_tstamp', 0)
                for hit in timing_ev.hits:
                    # hit = (channel, toa, tot)
                    toa = hit[1]
                    if toa > TOA_MAX_GLOBAL: continue

                    all_hits.append({
                        'time': ref+(toa*1E-3)/2, 
                        'toa': toa, 
                        'tot': hit[2],
                        'board': board_id, 
                        'ch': hit[0], 
                        'bank': bank_name
                    })

    if not all_hits: print("⚠️ Empty event (or all hits > TOA_MAX)."); return []
    all_hits.sort(key=lambda x: x['time'])
    
    sel = all_hits[:req_n_hits]
    print("\n--- Selected Hits ---")
    return sel


def yield_physics_events(filename, start_index=0, hits_per_event=10, 
                         toa_limits=(0, TOA_MAX_GLOBAL), tot_limits=(0, 1e9)):
    """MODE 3: Generator for sequential reading with CUTS."""
    try: mfile = midas.file_reader.MidasFile(filename)
    except Exception as e: print(f"File Error: {e}"); return
    
    phys_cnt = -1
    for event in mfile:
        if event.header.is_midas_internal_event(): continue
        if event.header.event_id != 1: continue
        
        phys_cnt += 1
        if phys_cnt < start_index: continue
        
        all_hits = []
        for bank_name, bank in event.banks.items():
            raw = bytes(bank.data); 
            if len(raw)<4: continue
            dtq = struct.unpack("<I", raw[:4])[0]
            if dtq == DTQ_TIMING:
                tev = Timing(raw)
                if tev.nhits > 0:
                    try: match = re.search(r'\d+', bank_name); bid = int(match.group()) if match else 0
                    except: bid=0
                    ref = getattr(tev, 'fine_tstamp', 0)
                    for h in tev.hits:
                        toa = h[1]
                        tot = h[2]
                        
                        # --- GLOBAL CUT ---
                        if toa > TOA_MAX_GLOBAL: continue
                        
                        # --- USER CUTS ---
                        if not (toa_limits[0] <= toa <= toa_limits[1]): continue
                        if not (tot_limits[0] <= tot <= tot_limits[1]): continue

                        all_hits.append({
                            'time': ref+(toa*1E-3)/2, 
                            'toa': toa, 
                            'tot': tot,
                            'board': bid, 
                            'ch': h[0],
                            'bank': bank_name
                        })
        
        if all_hits: all_hits.sort(key=lambda x: x['time'])
        
        sel = all_hits[:hits_per_event]
        yield phys_cnt, sel 


def read_cumulative_hits(filename):
    """MODE 2: Cumulative reading for heatmap."""
    print(f"\n--- STARTING CUMULATIVE ANALYSIS ON {filename} ---")
    try: mfile = midas.file_reader.MidasFile(filename)
    except Exception as e: print(f"Error: {e}"); return {}

    bundle_counts = {}
    cnt = 0
    for event in mfile:
        if event.header.is_midas_internal_event(): continue
        if event.header.event_id != 1: continue
        cnt += 1
        if cnt % 100 == 0: print(f"Processed {cnt} events...", end='\r')
        for bank_name, bank in event.banks.items():
            raw = bytes(bank.data)
            if len(raw) < 4: continue
            dtq = struct.unpack("<I", raw[:4])[0]
            if dtq == DTQ_TIMING:
                tev = Timing(raw)
                if tev.nhits > 0:
                    try: match = re.search(r'\d+', bank_name); bid = int(match.group()) if match else 0
                    except: bid = 0
                    for hit in tev.hits:
                        toa = hit[1]
                        if toa > TOA_MAX_GLOBAL: continue

                        b_id = get_bundle_id(bid, hit[0])
                        if b_id is not None: 
                            bundle_counts[b_id] = bundle_counts.get(b_id, 0) + 1
    
    print(f"\n✅ Finished. Total events: {cnt}. Active bundles: {len(bundle_counts)}")
    return bundle_counts

def analyze_toa_tot(filename):
    """MODE 4: ToA vs ToT 2D Histogram."""
    try:
        events_to_scan = int(input("\n>>> How many events to scan? "))
    except ValueError:
        print("Invalid input."); return

    print(f"\nScanning {events_to_scan} events in {filename}...")
    try: mfile = midas.file_reader.MidasFile(filename)
    except Exception as e: print(f"Error: {e}"); return

    toa_list = []
    tot_list = []
    
    count = 0
    for event in mfile:
        if event.header.is_midas_internal_event(): continue
        if event.header.event_id != 1: continue
        
        count += 1
        if count > events_to_scan: break
        if count % 100 == 0: print(f"Scanning event {count}...", end='\r')

        for bank_name, bank in event.banks.items():
            raw = bytes(bank.data)
            if len(raw) < 4: continue
            dtq = struct.unpack("<I", raw[:4])[0]
            
            if dtq == DTQ_TIMING:
                timing_ev = Timing(raw)
                for hit in timing_ev.hits:
                    toa = hit[1]
                    tot = hit[2]
                    
                    if toa > TOA_MAX_GLOBAL: continue
                    
                    toa_list.append(toa)
                    tot_list.append(tot)

    print(f"\nAnalysis complete. Found {len(toa_list)} total hits.")
    
    if len(toa_list) == 0:
        print("No hits found (or all filtered out).")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    h = ax.hist2d(toa_list, tot_list, bins=[200, 200], cmap='inferno', norm=LogNorm())
    cbar = plt.colorbar(h[3], ax=ax)
    cbar.set_label('Counts (Log Scale)')
    
    ax.set_title(f"ToA vs ToT ({count} Events) - ToA Cut < {TOA_MAX_GLOBAL}")
    ax.set_xlabel("Time of Arrival (ToA) [raw]")
    ax.set_ylabel("Time over Threshold (ToT) [raw]")
    plt.tight_layout()
    plt.show()

# ==========================================
# 1D. MANUAL DEBUG FUNCTION
# ==========================================
def debug_manual_mapping():
    def reverse_lookup_bundle(target_bundle):
        for b in range(4): 
            for c in range(64):
                if get_bundle_id(b, c) == target_bundle: return b, c
        return None, None

    print("\n================ MANUAL FIBER DEBUGGING ================")
    while True:
        print("\n1. Enter Board/Channel -> Get Bundle ID")
        print("2. Enter Bundle ID     -> Get Board/Channel")
        print("q. Quit")
        choice = input(">>> Choice: ").strip().lower()
        if choice == 'q': break
        try:
            bid = None
            if choice == '1':
                brd = int(input(">>> Board ID (0-3): "))
                chn = int(input(">>> Channel ID (0-63): "))
                bid = get_bundle_id(brd, chn)
                if bid is not None:
                    print(f"✅ MAPPING FOUND: Bundle {bid}")
                    title = f"DEBUG: Board {brd} Ch {chn} -> Bundle {bid}"
                else: print("❌ MAPPING NOT FOUND")
            elif choice == '2':
                bid = int(input(">>> Bundle ID: "))
                brd, chn = reverse_lookup_bundle(bid)
                if brd is not None:
                    print(f"✅ MAPPING FOUND: Board {brd} Ch {chn}")
                    title = f"DEBUG: Bundle {bid} -> Board {brd} Ch {chn}"
                else:
                    print("❌ MAPPING NOT FOUND"); bid=None
            
            if bid is not None:
                fig = mapper_plot_two_cylinders([bid])
                plt.suptitle(title, fontsize=14, fontweight='bold')
                plt.show()
        except ValueError: print("Invalid input.")

# ==========================================
# 2. MAPPING
# ==========================================
def get_bundle_id(board_id, channel_id):
    m0={31:0,27:1,23:2,19:3,15:4,11:5,7:6,3:7,29:8,25:9,21:10,17:11,13:12,9:13,5:14,1:15,0:16,4:17,8:18,12:19,16:20,20:21,24:22,28:23,2:24,6:25,10:26,14:27,18:28,22:29,26:30,30:31,32:32,36:33,40:34,44:35,48:36,52:37,56:38,60:39,34:40,38:41,42:42,46:43,50:44,54:45,58:46,62:47}
    m1={31:48,27:49,23:50,19:51,15:52,11:53,7:54,3:55,29:56,25:57,21:58,17:59,13:60,9:61,5:62,1:63,0:64,4:65,8:66,12:67,16:68,20:69,24:70,28:71,2:72,6:73,10:74,14:75,18:76,22:77,26:78,30:79,32:80,36:81,40:82,44:83,48:84,52:85,56:86,60:87,34:88,38:89,42:90,46:91,50:92,54:93}
    m2={31:0,27:1,23:2,19:3,15:4,11:5,7:6,3:7,29:8,25:9,21:10,17:11,13:12,9:13,5:14,1:15,0:16,4:17,8:18,12:19,16:20,20:21,24:22,28:23,2:24,6:25,10:26,14:27,18:28,22:29,26:30,30:31,63:32,59:33,55:34,51:35,47:36,43:37,39:38,35:39,61:40,57:41,53:42,49:43,45:44,41:45,37:46,33:47,32:48,36:49,40:50,44:51,48:52,52:53,56:54,60:55,34:56,38:57,42:58,46:59,50:60,54:61,58:62,62:63}
    m3={31:64,27:65,23:66,19:67,15:68,11:69,7:70,3:71,29:72,25:73,21:74,17:75,13:76,9:77,5:78,1:79,0:80,4:81,8:82,12:83,16:84,20:85,24:86,28:87,2:88,6:89,10:90,14:91,18:92,22:93,26:94,30:95,63:96,59:97,55:98,51:99,47:100,43:101,39:102,35:103,61:104,57:105,53:106,49:107,45:108,41:109,37:110,33:111,32:112,36:113,40:114,44:115,48:116,52:117,56:118}
    maps = [m0, m1, m2, m3]
    
    if board_id < 0 or board_id >= len(maps): return None
    val = maps[board_id].get(channel_id, None)
    if val is None: return None
    if board_id > 1: return val + (N1 + N2)
    else: return val

# ==========================================
# 3A. PLOTTING SINGLE EVENT (3D + 2D)
# ==========================================
def mapper_plot_two_cylinders(bundles_green, N1=N1, N2=N2, N3=N3, N4=N4, L=L_FIBER, event_idx=-1):
    fig = plt.figure(figsize=(18, 8))
    ax = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)
    ax2.set_aspect('equal')
    ax2.grid(True, alpha=0.3)
    R_C1, R_C2 = 1.7, 2.1
    ranges = [0, N1, N1+N2, N1+N2+N3, N1+N2+N3+N4]

    def get_geom(b):
        if b < ranges[1]: 
            return R_C1, 2*np.pi*(-b)/N1 + PHI_OFFSET_C1IN, -1, "red", 1
        elif b < ranges[2]: 
            b_loc = b - ranges[1]
            return R_C1, 2*np.pi*(b_loc)/N2 + PHI_OFFSET_C1OUT, 1, "lightsalmon", 1
        elif b < ranges[3]: 
            b_loc = b - ranges[2]
            return R_C2, 2*np.pi*(-b_loc)/N3 + PHI_OFFSET_C2IN, -1, "blue", 2
        elif b < ranges[4]: 
            b_loc = b - ranges[3]
            return R_C2, 2*np.pi*(b_loc)/N4 + PHI_OFFSET_C2OUT, 1, "deepskyblue", 2
        return None

    # Background
    for i in range(ranges[4]):
        g = get_geom(i)
        if g:
            z = np.linspace(-L, L, 20)
            phi = g[1] + g[2] * ((z+L)/(2*L))*np.pi
            ax.plot(z, g[0]*np.cos(phi), g[0]*np.sin(phi), lw=0.1, color=g[3])

    # Active
    c1cw, c1ccw, c2cw, c2ccw = [],[],[],[]
    for b in bundles_green:
        g = get_geom(b)
        if g:
            z = np.linspace(-L, L, 100)
            phi = g[1] + g[2] * ((z+L)/(2*L))*np.pi
            ax.plot(z, g[0]*np.cos(phi), g[0]*np.sin(phi), lw=2.5, color=g[3])
            
            pc = g[1] + g[2] * ((0+L)/(2*L))*np.pi
            x0, y0 = g[0]*np.cos(pc), g[0]*np.sin(pc)
            ax2.scatter(x0, y0, s=100, color=g[3], edgecolors='k', zorder=10)
            
            label = b if b < (N1+N2) else b - (N1+N2)
            txt_off = 1.15 if b < (N1+N2) else 1.10
            ax2.text(x0*txt_off, y0*txt_off, str(label), fontsize=9, ha='center', va='center')
            
            if g[4]==1: (c1cw if g[2]==-1 else c1ccw).append((b, g[1]))
            else:       (c2cw if g[2]==-1 else c2ccw).append((b, g[1]))

    def do_cross(cw, ccw, R):
        for b1,p1 in cw:
            for b2,p2 in ccw:
                delta = p1 - p2
                for k in range(-2, 3):
                    th = delta/2.0 - k*np.pi
                    if -0.001<=th<=np.pi+0.001:
                        ax.scatter((2*L*th/np.pi)-L, R*np.cos(p2+th), R*np.sin(p2+th), s=200, marker='*', color='gold', ec='k', zorder=20)
    do_cross(c1cw, c1ccw, R_C1)
    do_cross(c2cw, c2ccw, R_C2)

    ax.legend(handles=[Line2D([0],[0], color=c, lw=2, label=l) for c,l in zip(['red','lightsalmon','blue','deepskyblue'], ['C1 In','C1 Out','C2 In','C2 Out'])])
    
    title_str = "3D Event View"
    if event_idx >= 0: title_str += f" | Event #{event_idx}"
    ax.set_title(title_str); ax.view_init(elev=20, azim=-60)
    ax2.set_title("2D View"); ax2.add_artist(plt.Circle((0,0), R_C1, fill=False)); ax2.add_artist(plt.Circle((0,0), R_C2, fill=False, ls='--'))
    ax2.set_xlim(-3.5,3.5); ax2.set_ylim(-3.5,3.5)
    plt.tight_layout()
    return fig

# ==========================================
# 3B. PLOTTING HEATMAP
# ==========================================
def mapper_plot_heatmap(bundle_counts, N1=N1, N2=N2, N3=N3, N4=N4, L=L_FIBER):
    fig = plt.figure(figsize=(18, 9))
    gs = GridSpec(4, 2, width_ratios=[1, 1.2]) 
    
    ax_h1 = fig.add_subplot(gs[0, 0])
    ax_h2 = fig.add_subplot(gs[1, 0], sharey=ax_h1)
    ax_h3 = fig.add_subplot(gs[2, 0], sharey=ax_h1)
    ax_h4 = fig.add_subplot(gs[3, 0], sharey=ax_h1)
    ax2d = fig.add_subplot(gs[:, 1])
    ax2d.set_aspect('equal')

    data_c1_in = [bundle_counts.get(i, 0) for i in range(0, N1)]
    data_c1_out = [bundle_counts.get(i, 0) for i in range(N1, N1+N2)]
    idx3 = N1+N2
    data_c2_in = [bundle_counts.get(i, 0) for i in range(idx3, idx3+N3)]
    idx4 = idx3+N3
    data_c2_out = [bundle_counts.get(i, 0) for i in range(idx4, idx4+N4)]

    ax_h1.bar(range(0, N1), data_c1_in, color='red', alpha=0.7)
    ax_h1.set_title(f"Cyl 1 Inner - Tot: {sum(data_c1_in)}")
    ax_h1.grid(True, alpha=0.3)
    ax_h2.bar(range(0, N2), data_c1_out, color='lightsalmon', alpha=0.7)
    ax_h2.set_title(f"Cyl 1 Outer - Tot: {sum(data_c1_out)}")
    ax_h2.grid(True, alpha=0.3)
    ax_h3.bar(range(0, N3), data_c2_in, color='blue', alpha=0.7)
    ax_h3.set_title(f"Cyl 2 Inner - Tot: {sum(data_c2_in)}")
    ax_h3.grid(True, alpha=0.3)
    ax_h4.bar(range(0, N4), data_c2_out, color='deepskyblue', alpha=0.7)
    ax_h4.set_title(f"Cyl 2 Outer - Tot: {sum(data_c2_out)}")
    ax_h4.grid(True, alpha=0.3)
    ax_h4.set_xlabel("Bundle ID (Relative)")

    R_C1, R_C2 = 1.7, 2.1
    idx_start_c1_in, idx_start_c1_out = 0, N1
    idx_start_c2_in, idx_start_c2_out = N1+N2, N1+N2+N3
    total_fibers = N1+N2+N3+N4

    max_val = max(bundle_counts.values()) if bundle_counts else 1
    cmap = cm.cividis 
    norm = mcolors.Normalize(vmin=0, vmax=max_val)
    DR_VIS = 0.08

    def get_vis_geom(b_idx):
        if b_idx < idx_start_c1_out:
            phi = 2*np.pi*(-b_idx)/N1 + PHI_OFFSET_C1IN
            return R_C1 - DR_VIS, phi, -1
        elif b_idx < idx_start_c2_in:
            bl = b_idx - idx_start_c1_out
            phi = 2*np.pi*(bl)/N2 + PHI_OFFSET_C1OUT
            return R_C1 + DR_VIS, phi, 1
        elif b_idx < idx_start_c2_out:
            bl = b_idx - idx_start_c2_in
            phi = 2*np.pi*(-bl)/N3 + PHI_OFFSET_C2IN
            return R_C2 - DR_VIS, phi, -1
        elif b_idx < total_fibers:
            bl = b_idx - idx_start_c2_out
            phi = 2*np.pi*(bl)/N4 + PHI_OFFSET_C2OUT
            return R_C2 + DR_VIS, phi, 1
        return None

    for i in range(total_fibers):
        g = get_vis_geom(i)
        if g:
            R_vis, phi0, d = g
            phi_c = phi0 + d*((0+L)/(2*L))*np.pi
            x0, y0 = R_vis*np.cos(phi_c), R_vis*np.sin(phi_c)
            cnt = bundle_counts.get(i, 0)
            if cnt > 0:
                color = cmap(norm(cnt))
                alpha, ec, sz, zo = 1.0, 'k', 120, 20
            else:
                color = 'whitesmoke'; alpha, ec, sz, zo = 0.8, 'silver', 60, 5
            ax2d.scatter(x0, y0, s=sz, color=color, edgecolors=ec, alpha=alpha, zorder=zo)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    cb = plt.colorbar(sm, ax=ax2d, fraction=0.046, pad=0.04)
    cb.set_label('Hits (Cividis)')

    ax2d.set_title(f"Cumulative Hitmap (Tot: {sum(bundle_counts.values())})")
    ax2d.set_xlim(-3.5, 3.5); ax2d.set_ylim(-3.5, 3.5)
    ax2d.add_artist(plt.Circle((0,0), R_C1, fill=False, lw=0.5, ls='--'))
    ax2d.add_artist(plt.Circle((0,0), R_C2, fill=False, lw=0.5, ls='--'))
    plt.tight_layout(); plt.show()

# ==========================================
# 4. SEQUENTIAL RUNNER
# ==========================================
def run_sequential_mode(filename):
    try:
        start_idx = int(input("\n>>> Which physics event to start from? (0, 1...): "))
        hits_req = int(input(">>> How many hits to visualize per event? "))
        print("\n--- Optional Cuts (press Enter to skip) ---")
        t_min_in = input(f"Min ToA [0]: ")
        toa_min = int(t_min_in) if t_min_in else 0
        t_max_in = input(f"Max ToA [{TOA_MAX_GLOBAL}]: ")
        toa_max = int(t_max_in) if t_max_in else TOA_MAX_GLOBAL
        tot_min_in = input(f"Min ToT [0]: ")
        tot_min = int(tot_min_in) if tot_min_in else 0
        tot_max_in = input(f"Max ToT [No Limit]: ")
        tot_max = int(tot_max_in) if tot_max_in else 1000000000
    except ValueError:
        print("Invalid input."); return

    print(f"\n--- Starting Sequential Mode from event #{start_idx} ---")
    print(f"Cuts: ToA [{toa_min}-{toa_max}], ToT [{tot_min}-{tot_max}]")
    print("Press [ENTER] in console for next event.")
    print("Type 'q' and [ENTER] to quit.\n")
    
    # Create generator with cuts
    gen = yield_physics_events(filename, start_index=start_idx, hits_per_event=hits_req,
                               toa_limits=(toa_min, toa_max), tot_limits=(tot_min, tot_max))
    
    for ev_idx, hits_data in gen:
        print(f"\n========================================")
        print(f"--> Physics Event #{ev_idx}")
        print("========================================")
        print("--- Selected Hits ---")
        
        bids = []
        for i, h in enumerate(hits_data):
            print(f"{i+1}) Board: {h['board']:<2} | Ch: {h['ch']:<2} | ToA: {h['toa']} | ToT: {h['tot']} ({h['bank']})")
            bi = get_bundle_id(h['board'], h['ch'])
            if bi is not None: bids.append(bi)
            else: print(f"⚠️ WARNING: Mapping not found for Board {h['board']} Channel {h['ch']}")
        
        fig = mapper_plot_two_cylinders(bids, event_idx=ev_idx)
        plt.show(block=False); plt.pause(0.1) 
        if input("\n([Enter]=Next, 'q'=Quit) >>> ").lower().strip() == 'q': plt.close(fig); break
        plt.close(fig)
    print("End.")

# ==========================================
# 5. MAIN
# ==========================================
if __name__ == "__main__":
    
    # Check if filename is passed as argument
    if len(sys.argv) > 1:
        FILENAME = sys.argv[1]
        print(f"File loaded: {FILENAME}")
    else:
        FILENAME = input("Enter .mid.lz4 filename (e.g. ../run00200.mid.lz4): ").strip()
        if not FILENAME:
            print("No file provided. Exiting.")
            sys.exit(0)

    # Check file existence
    if not os.path.exists(FILENAME):
        print(f"ERROR: File '{FILENAME}' not found.")
        sys.exit(1)

    print("========================================")
    print("       MIDAS FIBER ANALYZER            ")
    print("========================================")
    print("1. Single Event (Specific)")
    print("2. Cumulative Histogram (Heatmap)")
    print("3. Sequential (Event by event)")
    print("4. ToA vs ToT 2D Histogram")
    print("5. Manual Debug (Check Mapping)")
    
    m = input("\n>>> Select mode (1-5): ")
    
    if m=='1':
        hits = read_nth_physics_event(FILENAME)
        if hits:
            bids = []
            for i, h in enumerate(hits):
                print(f"{i+1}) Board: {h['board']:<2} | Ch: {h['ch']:<2} | ToA: {h['toa']} | ToT: {h['tot']}")
                bi = get_bundle_id(h['board'], h['ch'])
                if bi is not None: bids.append(bi)
                else: print(f"⚠️ WARNING: Mapping not found for Board {h['board']} Channel {h['ch']}")
            mapper_plot_two_cylinders(bids).show(); plt.show()

    elif m=='2': mapper_plot_heatmap(read_cumulative_hits(FILENAME))
    elif m=='3': run_sequential_mode(FILENAME)
    elif m=='4': analyze_toa_tot(FILENAME)
    elif m=='5': debug_manual_mapping()
    else: print("Invalid choice.")
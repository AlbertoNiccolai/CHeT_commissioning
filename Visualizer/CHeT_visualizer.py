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
OFFSET_EXP = 40 * np.pi/180
DELTA1 = 1.26732381973286972
DELTA2 = 1.42022338200000001
PHI_OFFSET_C1IN  = 4.293507822806237 + DELTA1 + OFFSET_EXP
PHI_OFFSET_C1OUT = 3.829038665089601 + DELTA1 + OFFSET_EXP
PHI_OFFSET_C2IN  = 2.609122003 + DELTA2 + OFFSET_EXP
PHI_OFFSET_C2OUT = 3.351032122 + DELTA2 + OFFSET_EXP

TOA_MAX_GLOBAL = 500  # Hard cutoff for Time of Arrival (Noise filter)

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
        time0 = 1E18
        raw = bytes(bank.data)
        if len(raw) < 4: continue
        dtq = struct.unpack("<I", raw[:4])[0]
        if dtq == DTQ_TIMING:
            timing_ev = Timing(raw)
            if timing_ev.nhits > 0:
                try: match = re.search(r'\d+', bank_name); board_id = int(match.group()) if match else 0
                except: board_id = 0
                
                # --- FILTER: EXCLUDE BOARD >= 4 ---
                if board_id >= 4: continue
                # ----------------------------------

                ref = getattr(timing_ev, 'fine_tstamp', 0)
                for hit in timing_ev.hits:
                    toa = hit[1]
                    if board_id == 0 :
                        time0 = ref 
                    if toa > TOA_MAX_GLOBAL: continue

                    all_hits.append({
                        'time': ref +(toa*1E-3)/2, 
                        'time ref': ref,
                        'toa': toa,
                        'toa_correct': ref*1E3 +(toa)/2 - time0*1E3, 
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

def yield_physics_events(filename, start_index=0, toa_limits=(0, TOA_MAX_GLOBAL), tot_limits=(0, 1e9)):
    """MODE 3 Generator: Yields ALL hits for the event within global cuts."""
    try: mfile = midas.file_reader.MidasFile(filename)
    except Exception as e: print(f"File Error: {e}"); return
    
    phys_cnt = -1
    for event in mfile:
        time0 = 1E18
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

                    # --- FILTER: EXCLUDE BOARD >= 4 ---
                    if bid >= 4: continue
                    # ----------------------------------

                    ref = getattr(tev, 'fine_tstamp', 0)
                    if ref < time0:
                        time0 = ref
                    for h in tev.hits:
                        toa = h[1]
                        treference = ref*1E3+ toa/2 - time0*1E3 #ns
                        tot = h[2]
                        
                        # --- GLOBAL CUT ---
                        if treference > TOA_MAX_GLOBAL: continue
                        
                        # --- USER CUTS (Initial broad cuts) ---
                        if not (toa_limits[0] <= treference <= toa_limits[1]): continue
                        if not (tot_limits[0] <= tot <= tot_limits[1]): continue

                        all_hits.append({
                            'time': ref+(toa*1E-3)/2, 
                            'time ref': ref,
                            'toa': toa, 
                            'toa_correct': treference, 
                            'tot': tot,
                            'board': bid, 
                            'ch': h[0],
                            'bank': bank_name
                        })
        
        if all_hits: all_hits.sort(key=lambda x: x['time'])
        yield phys_cnt, all_hits

def read_cumulative_hits(filename, toa_limits=(0, TOA_MAX_GLOBAL), tot_limits=(0, 1e9)):
    """
    MODE 2: Cumulative reading.
    Returns:
    1. Bundle Counts (dict)
    2. Global ToA list
    3. Global ToT list
    4. Delta ToA per event list
    5. Delta ToT per event list
    6. Hit Multiplicity Data (dict with lists of counts per event)
    7. ToA by Board (dict)
    8. ToT by Board (dict)
    """
    print(f"\n--- STARTING CUMULATIVE ANALYSIS ON {filename} ---")
    print(f"Applying Cuts -> ToA Corrected: {toa_limits}, ToT: {tot_limits}")
    print("⚠️  FILTER ACTIVE: Skipping Boards >= 4")
    
    try: mfile = midas.file_reader.MidasFile(filename)
    except Exception as e: print(f"Error: {e}"); return {}, [], [], [], [], {}, {}, {}

    bundle_counts = {}

    # LISTE PER LE DISTRIBUZIONI GLOBALI (Hit singole)
    dist_toa_all = []       
    dist_tot_all = []        
    dist_delta_toa = []      
    dist_delta_tot = []

    # NEW: LISTE PER BOARD
    dist_toa_by_board = {}
    dist_tot_by_board = {}

    # LISTE PER IL CONTEGGIO HIT PER EVENTO
    hits_ev_total = []
    hits_ev_c1 = []
    hits_ev_c2 = []
    hits_ev_l1 = []
    hits_ev_l2 = []
    hits_ev_l3 = []
    hits_ev_l4 = []

    # Indici per i layer
    idx_l2 = N1
    idx_l3 = N1 + N2
    idx_l4 = N1 + N2 + N3
    idx_end = N1 + N2 + N3 + N4

    cnt = 0
    for event in mfile:
        time0 = 1E18
        if event.header.is_midas_internal_event(): continue
        if event.header.event_id != 1: continue
        cnt += 1
        if cnt % 100 == 0: print(f"Processed {cnt} events...", end='\r')
        
        # Liste temporanee per calcolare il delta DELL'EVENTO CORRENTE
        event_valid_toas = []
        event_valid_tots = []
        
        # Contatori hit per questo evento
        c_tot = 0
        c_c1, c_c2 = 0, 0
        c_l1, c_l2, c_l3, c_l4 = 0, 0, 0, 0

        for bank_name, bank in event.banks.items():
            raw = bytes(bank.data)
            if len(raw) < 4: continue
            dtq = struct.unpack("<I", raw[:4])[0]
            if dtq == DTQ_TIMING:
                tev = Timing(raw)
                ref = getattr(tev, 'fine_tstamp', 0)
                if ref < time0: time0 = ref 
                
                if tev.nhits > 0:
                    try: match = re.search(r'\d+', bank_name); bid = int(match.group()) if match else 0
                    except: bid = 0
                    
                    # --- FILTER: EXCLUDE BOARD >= 4 ---
                    if bid >= 4: continue
                    # ----------------------------------

                    for hit in tev.hits:
                        toa = hit[1]
                        tot = hit[2]
                        treference = ref*1E3+ toa/2 - time0*1E3 

                        # Global Cut on Noise
                        if toa > TOA_MAX_GLOBAL: continue

                        # User Cuts
                        if not (toa_limits[0] <= treference <= toa_limits[1]): continue
                        if not (tot_limits[0] <= tot <= tot_limits[1]): continue
                        
                        # --- Hit accettata ---
                        dist_toa_all.append(treference)
                        dist_tot_all.append(tot)
                        event_valid_toas.append(treference)
                        event_valid_tots.append(tot)

                        # --- Salvataggio per Board ---
                        if bid not in dist_toa_by_board: dist_toa_by_board[bid] = []
                        if bid not in dist_tot_by_board: dist_tot_by_board[bid] = []
                        dist_toa_by_board[bid].append(treference)
                        dist_tot_by_board[bid].append(tot)

                        # Mappatura e Conteggi
                        b_id = get_bundle_id(bid, hit[0])
                        if b_id is not None: 
                            bundle_counts[b_id] = bundle_counts.get(b_id, 0) + 1
                            c_tot += 1
                            
                            # Logica Layers/Cilindri
                            if b_id < idx_l2:      # Layer 1 (C1 In)
                                c_l1 += 1; c_c1 += 1
                            elif b_id < idx_l3:    # Layer 2 (C1 Out)
                                c_l2 += 1; c_c1 += 1
                            elif b_id < idx_l4:    # Layer 3 (C2 In)
                                c_l3 += 1; c_c2 += 1
                            elif b_id < idx_end:   # Layer 4 (C2 Out)
                                c_l4 += 1; c_c2 += 1

        # Fine evento: Calcolo delta e salvataggio conteggi
        if event_valid_toas:
            dist_delta_toa.append(max(event_valid_toas) - min(event_valid_toas))
        if event_valid_tots:
            dist_delta_tot.append(max(event_valid_tots) - min(event_valid_tots))
        
        # Salviamo i contatori anche se sono 0 (per statistica)
        hits_ev_total.append(c_tot)
        hits_ev_c1.append(c_c1)
        hits_ev_c2.append(c_c2)
        hits_ev_l1.append(c_l1); hits_ev_l2.append(c_l2)
        hits_ev_l3.append(c_l3); hits_ev_l4.append(c_l4)

    print(f"\n✅ Finished. Total events: {cnt}. Active bundles: {len(bundle_counts)}")
    print(f"   Hits passing cuts: {len(dist_toa_all)}")

    hit_counts_data = {
        'total': hits_ev_total,
        'c1': hits_ev_c1, 'c2': hits_ev_c2,
        'l1': hits_ev_l1, 'l2': hits_ev_l2, 'l3': hits_ev_l3, 'l4': hits_ev_l4
    }

    return bundle_counts, dist_toa_all, dist_tot_all, dist_delta_toa, dist_delta_tot, hit_counts_data, dist_toa_by_board, dist_tot_by_board

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
        time0 = 1E10
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
                try: match = re.search(r'\d+', bank_name); bid = int(match.group()) if match else 0
                except: bid=0
                
                # --- FILTER: EXCLUDE BOARD >= 4 ---
                if bid >= 4: continue
                # ----------------------------------

                ref = getattr(timing_ev, 'fine_tstamp', 0)
                if ref < time0: 
                    time0 = ref
                for hit in timing_ev.hits:
                    toa = hit[1]
                    treference = ref*1E3+ toa/2 - time0*1E3 #ns
                    tot = hit[2]
                    
                    if treference > TOA_MAX_GLOBAL: continue
                    
                    toa_list.append(treference)
                    tot_list.append(tot)

    print(f"\nAnalysis complete. Found {len(toa_list)} total hits.")

    if len(toa_list) == 0:
        print("No hits found (or all filtered out).")
        return

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))
    h = ax.hist2d(toa_list, tot_list, bins=[300, 300], cmap='inferno_r', norm=LogNorm())
    cbar = plt.colorbar(h[3], ax=ax)
    cbar.set_label('Counts (Log Scale)')

    ax.set_title(f"(ToA + Tref - T0) vs ToT ({count} Events) - ToA Cut < {TOA_MAX_GLOBAL}")
    ax.set_xlabel("Time of arrival wrt to T ref [ns]")
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
    
    print("\n================ MANUAL FIBER DEBUGGING (MULTI) ================")
    while True:
        print("\n1. Enter list of 'Board,Channel' -> Get Bundle IDs")
        print("2. Enter list of Bundle IDs      -> Get Board/Channels")
        print("q. Quit")
        choice = input(">>> Choice: ").strip().lower()
        if choice == 'q': break
        bundles_to_plot = []
        try:
            if choice == '1':
                raw_input = input(">>> Enter pairs (e.g. 0,10 0,11): ").strip()
                tokens = raw_input.replace(';', ' ').split()
                for token in tokens:
                    try:
                        if ',' not in token: continue
                        b_str, c_str = token.split(',')
                        bid = get_bundle_id(int(b_str), int(c_str))
                        if bid is not None: bundles_to_plot.append(bid)
                    except ValueError: pass
                title = f"DEBUG: {len(bundles_to_plot)} Bundles"

            elif choice == '2':
                raw_input = input(">>> Enter Bundle IDs (e.g. 100 101): ").strip()
                tokens = raw_input.replace(',', ' ').split()
                for token in tokens:
                    try:
                        bid = int(token)
                        brd, chn = reverse_lookup_bundle(bid)
                        if brd is not None: bundles_to_plot.append(bid)
                    except ValueError: pass
                title = f"DEBUG: {len(bundles_to_plot)} Bundles"
            else: continue

            if bundles_to_plot:
                fig = mapper_plot_two_cylinders(bundles_to_plot)
                plt.suptitle(title, fontsize=14, fontweight='bold')
                plt.show()
        except Exception as e: print(f"Error: {e}")

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
# 3. PLOTTING
# ==========================================

def plot_run_distributions(toa_list, tot_list, delta_toa_list, delta_tot_list, cuts_info):
    """Plot global histograms: Single hits (Row 1) and Event Deltas (Row 2)."""
    if not toa_list:
        print("⚠️ No data to plot for distributions.")
        return
    
    # Helper per calcolare i bin dinamici
    def get_dynamic_bins(data_list):
        if not data_list: return 100
        mn, mx = min(data_list), max(data_list)
        rng = mx - mn
        if rng <= 0: return 100
        nbins = int(2*rng)
        if nbins < 10: nbins = 10
        if nbins > 120: nbins = 120
        return nbins

    # Crea griglia 2x2
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))

    # --- ROW 1: Single Hit Distributions ---
    
    # Plot ToA Corrected
    b_toa = get_dynamic_bins(toa_list)
    ax[0, 0].hist(toa_list, bins=b_toa, color='teal', alpha=0.7, edgecolor='k', linewidth=0.5)
    ax[0, 0].set_title(f"Global Corrected ToA (Single Hits)\n(Cut: {cuts_info['toa']}) - Bins: {b_toa}")
    ax[0, 0].set_xlabel("ToA Corrected (Ref + ToA/2 - T0) [ns]")
    ax[0, 0].set_ylabel("Counts")
    ax[0, 0].grid(True, alpha=0.3)
    ax[0, 0].set_yscale('log')

    # Plot ToT
    b_tot = get_dynamic_bins(tot_list)
    ax[0, 1].hist(tot_list, bins=b_tot, color='orange', alpha=0.7, edgecolor='k', linewidth=0.5)
    ax[0, 1].set_title(f"Global ToT (Single Hits)\n(Cut: {cuts_info['tot']}) - Bins: {b_tot}")
    ax[0, 1].set_xlabel("Time over Threshold (ToT) [raw]")
    ax[0, 1].set_ylabel("Counts")
    ax[0, 1].grid(True, alpha=0.3)
    ax[0, 1].set_yscale('log')

    # --- ROW 2: Event Delta Distributions (Max - Min) ---
    
    # Plot Delta ToA
    b_dtoa = get_dynamic_bins(delta_toa_list)
    ax[1, 0].hist(delta_toa_list, bins=b_dtoa, color='darkgreen', alpha=0.7, edgecolor='k', linewidth=0.5)
    ax[1, 0].set_title(f"Event ToA Range (Max - Min)\nEvents with >0 hits - Bins: {b_dtoa}")
    ax[1, 0].set_xlabel("Delta ToA [ns]")
    ax[1, 0].set_ylabel("Events")
    ax[1, 0].grid(True, alpha=0.3)
    ax[1, 0].set_yscale('log')

    # Plot Delta ToT
    b_dtot = get_dynamic_bins(delta_tot_list)
    ax[1, 1].hist(delta_tot_list, bins=b_dtot, color='darkred', alpha=0.7, edgecolor='k', linewidth=0.5)
    ax[1, 1].set_title(f"Event ToT Range (Max - Min)\nEvents with >0 hits - Bins: {b_dtot}")
    ax[1, 1].set_xlabel("Delta ToT [raw]")
    ax[1, 1].set_ylabel("Events")
    ax[1, 1].grid(True, alpha=0.3)
    ax[1, 1].set_yscale('log')

    plt.tight_layout()
    print("\n>>> Displaying Global Distributions...")
    plt.show()

def plot_distributions_by_board(toa_by_board, tot_by_board):
    """
    Plots ToA and ToT histograms separated by Board ID.
    Ignores boards excluded by reading function (>=4).
    USES SAME BINNING FOR ALL BOARDS.
    """
    boards = sorted(toa_by_board.keys())
    if not boards:
        print("⚠️ No board data found to plot.")
        return

    # --- 1. Calcolo Binning Globale per ToA ---
    # Raccogliamo tutti i ToA di tutte le board per trovare min e max globali
    all_toas = []
    for b in boards:
        all_toas.extend(toa_by_board[b])
    
    if all_toas:
        g_min_toa, g_max_toa = min(all_toas), max(all_toas)
        rng_toa = g_max_toa - g_min_toa
        # Logica: circa 2 bin per ns, ma limitato tra 50 e 200 bin totali
        nbins_toa = int(rng_toa * 2)
        nbins_toa = max(50, min(nbins_toa, 200))
        # Creiamo i bordi esatti dei bin
        bins_toa_edges = np.linspace(g_min_toa, g_max_toa, nbins_toa)
    else:
        bins_toa_edges = 50 # Fallback

    # --- 2. Calcolo Binning Globale per ToT ---
    all_tots = []
    for b in boards:
        all_tots.extend(tot_by_board[b])
        
    if all_tots:
        g_min_tot, g_max_tot = min(all_tots), max(all_tots)
        rng_tot = g_max_tot - g_min_tot
        nbins_tot = int(rng_tot)
        nbins_tot = max(50, min(nbins_tot, 200))
        bins_tot_edges = np.linspace(g_min_tot, g_max_tot, nbins_tot)
    else:
        bins_tot_edges = 50

    n_boards = len(boards)
    # sharex='col' allinea gli assi X verticalmente per confronto diretto
    fig, axs = plt.subplots(n_boards, 2, figsize=(12, 3.5 * n_boards), squeeze=False, sharex='col')
    
    fig.suptitle(f"Distributions by Board (Filtered: Excluded Boards >= 4)", fontsize=16)

    for i, bid in enumerate(boards):
        # Dati
        toas = toa_by_board[bid]
        tots = tot_by_board[bid]
        
        # --- Colonna 0: ToA ---
        ax_toa = axs[i, 0]
        if toas:
            # Usiamo bins=bins_toa_edges invece di un intero
            ax_toa.hist(toas, bins=bins_toa_edges, color='teal', alpha=0.7, edgecolor='k', linewidth=0.5)
            ax_toa.set_yscale('log')
        
        ax_toa.set_title(f"Board {bid} - ToA Corrected ({len(toas)} hits)")
        ax_toa.set_ylabel("Counts")
        ax_toa.grid(True, alpha=0.3)
        if i == n_boards - 1: ax_toa.set_xlabel("Time [ns]")

        # --- Colonna 1: ToT ---
        ax_tot = axs[i, 1]
        if tots:
            # Usiamo bins=bins_tot_edges
            ax_tot.hist(tots, bins=bins_tot_edges, color='orange', alpha=0.7, edgecolor='k', linewidth=0.5)
            ax_tot.set_yscale('log')
            
        ax_tot.set_title(f"Board {bid} - ToT")
        ax_tot.grid(True, alpha=0.3)
        if i == n_boards - 1: ax_tot.set_xlabel("ToT [raw]")

    plt.tight_layout(rect=[0, 0.03, 1, 0.97]) 
    print("\n>>> Displaying Per-Board Distributions (Uniform Bins)...")
    plt.show()
def plot_hit_multiplicity(counts_data):
    """
    Plots the multiplicity of hits per event:
    1. Total hits per event
    2. Comparison between Cylinder 1 and Cylinder 2
    3. Comparison between the 4 Layers
    """
    if not counts_data['total']:
        print("⚠️ No multiplicity data to plot.")
        return
    
    fig = plt.figure(figsize=(15, 9))
    gs = GridSpec(2, 2)
    
    # 1. Total Hits
    ax1 = fig.add_subplot(gs[0, :]) # Full width top
    max_h = max(counts_data['total'])
    bins = range(0, max_h + 2)
    ax1.hist(counts_data['total'], bins=bins, color='black', alpha=0.6, edgecolor='k', label='Total')
    ax1.set_title(f"Total Hits per Event (Max: {max_h})")
    ax1.set_xlabel("Number of Hits")
    ax1.set_ylabel("Count")
    ax1.set_yscale('log')
    ax1.grid(True, which="both", ls="-", alpha=0.2)
    
    # 2. Cylinders Comparison
    ax2 = fig.add_subplot(gs[1, 0])
    max_c = max(max(counts_data['c1']), max(counts_data['c2']))
    bins_c = range(0, max_c + 2)
    ax2.hist(counts_data['c1'], bins=bins_c, color='red', alpha=0.5, label='Cyl 1', histtype='stepfilled', edgecolor='red')
    ax2.hist(counts_data['c2'], bins=bins_c, color='blue', alpha=0.5, label='Cyl 2', histtype='stepfilled', edgecolor='blue')
    ax2.set_title("Hits per Cylinder")
    ax2.set_xlabel("Number of Hits")
    ax2.set_ylabel("Count")
    ax2.set_yscale('log')
    ax2.legend()
    ax2.grid(True, which="both", ls="-", alpha=0.2)
    
    # 3. Layers Comparison
    ax3 = fig.add_subplot(gs[1, 1])
    max_l = max([max(counts_data[k]) for k in ['l1','l2','l3','l4']])
    bins_l = range(0, max_l + 2)

    ax3.hist(counts_data['l1'], bins=bins_l, histtype='step', lw=2, color='red', label='L1 (C1 In)')
    ax3.hist(counts_data['l2'], bins=bins_l, histtype='step', lw=2, color='salmon', label='L2 (C1 Out)')
    ax3.hist(counts_data['l3'], bins=bins_l, histtype='step', lw=2, color='blue', label='L3 (C2 In)')
    ax3.hist(counts_data['l4'], bins=bins_l, histtype='step', lw=2, color='cyan', label='L4 (C2 Out)')

    ax3.set_title("Hits per Layer")
    ax3.set_xlabel("Number of Hits")
    ax3.set_ylabel("Count")
    ax3.set_yscale('log')
    ax3.legend()
    ax3.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    print("\n>>> Displaying Hit Multiplicity plots...")
    plt.show()

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
            return R_C1, 2*np.pi*(+b_loc)/N2 + PHI_OFFSET_C1OUT, 1, "lightsalmon", 1
        elif b < ranges[3]: 
            b_loc = b - ranges[2]
            return R_C2, 2*np.pi*(-b_loc)/N3 + PHI_OFFSET_C2IN, -1, "blue", 2
        elif b < ranges[4]: 
            b_loc = b - ranges[3]
            return R_C2, 2*np.pi*(+b_loc)/N4 + PHI_OFFSET_C2OUT, 1, "deepskyblue", 2
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
            z_0 = 15 # Nota che qui vediamo la posizione downstream!
            pc = g[1] + g[2] * ((z_0+L)/(2*L))*np.pi
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

    ax.text(-L, 0, 0, "US", color='k', fontsize=16)
    ax.text(+L, 0, 0, "DS", color='k', fontsize=16)

    ax.legend(handles=[Line2D([0],[0], color=c, lw=2, label=l) for c,l in zip(['red','lightsalmon','blue','deepskyblue'], ['C1 In','C1 Out','C2 In','C2 Out'])])

    title_str = "3D Event View"
    if event_idx >= 0: title_str += f" | Event #{event_idx}"
    ax.set_title(title_str); ax.view_init(elev=20, azim=-60)
    ax2.set_title("2D View"); ax2.add_artist(plt.Circle((0,0), R_C1, fill=False)); ax2.add_artist(plt.Circle((0,0), R_C2, fill=False))
    ax2.set_xlim(-3.5,3.5); ax2.set_ylim(-3.5,3.5)
    plt.tight_layout()
    return fig

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
    cmap = cm.inferno.reversed() 
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
    cb.set_label('Hits')

    ax2d.set_title(f"Cumulative Hitmap (Tot: {sum(bundle_counts.values())})")
    ax2d.set_xlim(-3.5, 3.5); ax2d.set_ylim(-3.5, 3.5)
    ax2d.add_artist(plt.Circle((0,0), R_C1, fill=False, lw=0.5, ls='--'))
    ax2d.add_artist(plt.Circle((0,0), R_C2, fill=False, lw=0.5, ls='--'))
    plt.tight_layout(); plt.show()

# ==========================================
# 4. SEQUENTIAL RUNNER
# ==========================================

def plot_event_toa_histogram(hits, ev_idx):
    """Plots histogram of ToA for the current event (blocking)."""
    toas = [h['toa_correct'] for h in hits]
    if not toas:
        print("No ToA data to plot.")
        return
    
    plt.figure(figsize=(8, 5))
    plt.hist(toas, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f"Event #{ev_idx}: ToA Distribution")
    plt.xlabel("ToA corrected (raw)")
    plt.ylabel("Counts")
    plt.grid(True, alpha=0.3)
    print("\n>>> Displaying ToA Histogram. Close the plot window to continue...")
    plt.show()

def run_sequential_mode(filename):
    try:
        start_idx = int(input("\n>>> Which physics event to start from? (0, 1...): "))
        hits_req = 10000000
        print("\n--- Initial Global Cuts (press Enter to skip) ---")
        t_min_in = input(f"Min ToA [0]: ")
        toa_min_g = int(t_min_in) if t_min_in else 0
        t_max_in = input(f"Max ToA [{TOA_MAX_GLOBAL}]: ")
        toa_max_g = int(t_max_in) if t_max_in else TOA_MAX_GLOBAL
        tot_min_in = input(f"Min ToT [0]: ")
        tot_min_g = int(tot_min_in) if tot_min_in else 0
        tot_max_in = input(f"Max ToT [No Limit]: ")
        tot_max_g = int(tot_max_in) if tot_max_in else 1000000000
    except ValueError:
        print("Invalid input."); return
    
    print(f"\n--- Starting Sequential Mode from event #{start_idx} ---")
    gen = yield_physics_events(filename, start_index=start_idx, 
                               toa_limits=(toa_min_g, toa_max_g), 
                               tot_limits=(tot_min_g, tot_max_g))
    
    for ev_idx, hits_data in gen:
        print(f"\n========================================")
        print(f"--> Physics Event #{ev_idx} (Total Hits in range: {len(hits_data)})")
        print("========================================")
        plot_event_toa_histogram(hits_data, ev_idx)
        print(f"Current Range: {toa_min_g} - {toa_max_g}")
        cut_in = input(">>> Enter specific ToA cut (e.g. '100 200') or [Enter] to keep all: ").strip()
        local_hits = hits_data
        if cut_in:
            try:
                parts = cut_in.split()
                if len(parts) >= 2:
                    l_min, l_max = int(parts[0]), int(parts[1])
                    local_hits = [h for h in hits_data if l_min <= h['toa_correct'] <= l_max]
                    print(f"--> Applied Cut [{l_min}, {l_max}]. Hits remaining: {len(local_hits)}")
                else: print("⚠️ Invalid format.")
            except ValueError: print("⚠️ Invalid numbers.")
        
        sel_hits = local_hits[:hits_req]
        print("\n--- Visualizing Hits ---")
        bids = []
        for i, h in enumerate(sel_hits):
            # Check just to be safe, though filtered in generator
            if (h['board'] < 4):
                print(f"{i+1}) Board: {h['board']:<2} | Ch: {h['ch']:<2} | ToA: {h['toa']} | ToT: {h['tot']} | T ref: {h['time ref']} | ToA corrected: {h['toa_correct']} | ({h['bank']})")
            bi = get_bundle_id(h['board'], h['ch'])
            if bi is not None: bids.append(bi)
            else: print(f"⚠️ WARNING: Mapping not found for Board {h['board']} Channel {h['ch']}")
        
        fig = mapper_plot_two_cylinders(bids, event_idx=ev_idx)
        plt.show(block=False) 
        plt.pause(0.1) 
        user_cmd = input("\n([Enter]=Next, 'q'=Quit) >>> ").lower().strip()
        plt.close(fig)
        if user_cmd == 'q': break
    print("End.")

# ==========================================
# 5. MAIN
# ==========================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        FILENAME = sys.argv[1]
        print(f"File loaded: {FILENAME}")
    else:
        FILENAME = input("Enter .mid.lz4 filename (e.g. ../run00200.mid.lz4): ").strip()
        if not FILENAME:
            print("No file provided. Exiting.")
            sys.exit(0)
    
    if not os.path.exists(FILENAME):
        print(f"ERROR: File '{FILENAME}' not found.")
        sys.exit(1)

    print("========================================")
    print("       MIDAS FIBER ANALYZER            ")
    print("========================================")
    print("1. Single Event (Specific)")
    print("2. Cumulative Histogram (Heatmap & Cuts Analysis)")
    print("3. Sequential (Hist -> Cut -> Plot)")
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

    elif m=='2': 
        print("\n--- Optional Cuts for Cumulative Analysis ---")
        try:
            t_min_in = input(f"Min Corrected ToA [0]: ")
            toa_min = int(t_min_in) if t_min_in else 0
            t_max_in = input(f"Max Corrected ToA [{TOA_MAX_GLOBAL}]: ")
            toa_max = int(t_max_in) if t_max_in else TOA_MAX_GLOBAL
            
            tot_min_in = input(f"Min ToT [0]: ")
            tot_min = int(tot_min_in) if tot_min_in else 0
            tot_max_in = input(f"Max ToT [No Limit]: ")
            tot_max = int(tot_max_in) if tot_max_in else 1000000000
            
            # Recupera conteggi, liste distribuzioni e DIZIONARI PER BOARD
            counts, list_toa, list_tot, list_delta_toa, list_delta_tot, list_hits_data, dict_toa_board, dict_tot_board = read_cumulative_hits(
                FILENAME, 
                toa_limits=(toa_min, toa_max), 
                tot_limits=(tot_min, tot_max)
            )
            
            # 1. Plot Heatmap
            mapper_plot_heatmap(counts)
            
            # 2. Plot Distribuzioni Globali
            cuts_str = {'toa': f"{toa_min}-{toa_max}", 'tot': f"{tot_min}-{tot_max}"}
            plot_run_distributions(list_toa, list_tot, list_delta_toa, list_delta_tot, cuts_str)
            
            # 3. Plot Distribuzioni PER BOARD
            plot_distributions_by_board(dict_toa_board, dict_tot_board)

            # 4. Plot Hit Multiplicity
            plot_hit_multiplicity(list_hits_data)

        except ValueError:
            print("Invalid input.")

    elif m=='3': run_sequential_mode(FILENAME)
    elif m=='4': analyze_toa_tot(FILENAME)
    elif m=='5': debug_manual_mapping()
    else: print("Invalid choice.")
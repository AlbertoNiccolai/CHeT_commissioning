#Basic imports
from matplotlib import pyplot as plt
import midas.file_reader
import numpy as np
import os
import ROOT
import struct
import pprint

#This Function must be in the same folder as this code
from AcquisitionModes import Spectroscopy, Timing, Counting, Service
from AcquisitionModes import DTQ_SPECT, DTQ_TSPECT, DTQ_TIMING, DTQ_COUNT, DTQ_SERVICE
from channelsToBundles import channel_bundle_map_board0, channel_bundle_map_board1, channel_bundle_map_board2, channel_bundle_map_board3
from channelsToBundles import get_bundle_0, get_bundle_1, get_bundle_2, get_bundle_3
#from mapper_midas import mapper_plot
#from c1mapper import mapper_plot_c1
from c2mapper import mapper_plot_c2
from chetmapper import plot_cylinder




#LINEAR FUNCTIONS FOR 2D RECONSTRUCTION OF CYLINDER (not used)
#==============================================================
def linear_1_1(x, m, i1, N1 = 45., L = 15.):
    return L + m*(x-2*np.pi*i1/N1)

def linear_1_2(x, m, i1, N1 = 45., L = 15.):
    return L + m*(x-2*np.pi*i1/N1 - 2*np.pi)

def linear_2_1(x, m, i2, N2 = 49., L = 15.):
    return L - m*(x-2*np.pi*i2/N2)

def linear_2_2(x, m, i2, N2 = 49., L = 15.):
    return L - m*(x-2*np.pi*i2/N2 + 2*np.pi)
#==============================================================

#GLOBAL VARIABLES (lenght of cylinder, angle of fibers, number of fibers per layer)
#==============================================================)
m = 30. / np.pi
N1 = 45
N2 = 49
#==============================================================)
z = 13.5 
phi = 0.
phi = phi * np.pi/180.
#==============================================================)




#OPENING MIDAS FILE
#==============================================================)
mfile = midas.file_reader.MidasFile("/Users/Alberto/Downloads/run00200.mid.lz4")

try:
    odb = mfile.get_bor_odb_dump()
    odb_dict = odb.data
    print("From the ODB, AcqMode:")
    print(odb_dict["Equipment"]["FersDAQ"]["Settings"]["FersDAQ"]["FLab0"]["AcqMode"])
except RuntimeError:
    print("No ODB found")


mfile.jump_to_start()

all_events = []

for event in mfile:
    if event.header.is_midas_internal_event():
        continue
    if event.header.event_id != 1:
        continue

    event_record = {
        "serial": event.header.serial_number,
        "hits": []
    }
    for bank_name, bank in event.banks.items():
        raw = bytes(bank.data)
        dtq = struct.unpack("<I", raw[:4])[0]

        if dtq == DTQ_TIMING:
            timing_ev = Timing(raw)
            board_id = int(bank_name[-2:], 16)

            for hit in timing_ev.hits:
                ch   = hit["channel"]
                toa  = hit["tstamp"]
                tot  = hit["tot"]

                # Filtri? Falli qui.
                if tot <= 0:
                    continue

                # Bundle
                if board_id == 0:
                    bundle = get_bundle_0(ch)
                    if bundle == None:
                        bundle = -1
                elif board_id == 1:
                    bundle = get_bundle_1(ch)
                    if bundle == None:
                        bundle = -1
                elif board_id == 3:
                    bundle = get_bundle_2(ch)
                    if bundle == None:
                        bundle = -1
                elif board_id == 4:
                    bundle = get_bundle_3(ch)
                    if bundle == None:
                        bundle = -1
                else:
                    continue

                # Registra il singolo hit
                hit_info = {
                    "board_id": board_id,
                    "channel": ch,
                    "toa": toa,
                    "tot": tot,
                    "bundle": bundle,
                }

                event_record["hits"].append(hit_info)
    event_record["nhits"] = len(event_record["hits"])
    # Solo se l’evento ha hit significativi
    if len(event_record["hits"]) > 0:
        all_events.append(event_record)

#GET THE BUNDLE HITS DISTIBUTION (ALL, B0, B1)
#================================================
all_bundles = [
    hit["bundle"]
    for event in all_events
    for hit in event["hits"]
]
bundles_fired_b0 = [
    hit["bundle"]
    for event in all_events
    for hit in event["hits"]
    if hit["board_id"] == 0
]
bundles_fired_b1 = [
    hit["bundle"]
    for event in all_events
    for hit in event["hits"]
    if hit["board_id"] == 1
]
bundles_fired_b2 = [
    hit["bundle"]
    for event in all_events
    for hit in event["hits"]
    if hit["board_id"] == 2
]
bundles_fired_b3 = [
    hit["bundle"]
    for event in all_events
    for hit in event["hits"]
    if hit["board_id"] == 3
]
pprint.pprint(all_events[2])
print(f"I bundles di B2 sono: {bundles_fired_b2}")
#================================================



#GET TOA AND TOT FOR PLOTTING
#================================================
toa_fired = [
    hit["toa"]
    for event in all_events
    for hit in event["hits"]
]
tot_fired = [
    hit["tot"]
    for event in all_events
    for hit in event["hits"]
]
#================================================
#GET TOA AND TOT FOR PLOTTING, after CUT
#================================================
toa_fired_cut = [
    hit["toa"]
    for event in all_events
    for hit in event["hits"]
    if hit["toa"] < 550 and hit["toa"] > 250
]
tot_fired_cut = [
    hit["tot"]
    for event in all_events
    for hit in event["hits"]
    if hit["toa"] < 550 and hit["toa"] > 250
]
#================================================




#GET THE SELECTED EVENTS FOR MAPPERPLOT - CHOOSE A LOGIC
selected_events = [
    event
    for event in all_events
    if event["nhits"] >=1
    #and sum(1 for hit in event["hits"] if hit["bundle"] < 45) == 2
    #and sum(1 for hit in event["hits"] if hit["bundle"] >= 45) == 2
    #and max(hit["toa"] for hit in event["hits"]) - min(hit["toa"] for hit in event["hits"]) < 20
] #4 hits, 2 per layer, in time coincidence < 10 ns (20 lsb)

toa_selected_events = [
    hit["toa"]
    for event in selected_events
    for hit in event["hits"]
]
tot_selected_events = [
    hit["tot"]
    for event in selected_events
    for hit in event["hits"]
]





    #mapper_plot(bundle_list)



#print(selected_events)

for event in selected_events:
    if event["serial"] >= 10:
        continue  # salta questo evento

    bundle_list = [hit["bundle"] for hit in event["hits"]]
    bundles_green_C1 = [hit["bundle"] for hit in event["hits"] if (hit["board_id"] == 0 or hit["board_id"] == 1) and hit["tot"]>50]
    bundles_green_C2 = [hit["bundle"] for hit in event["hits"] if (hit["board_id"] == 2 or hit["board_id"] == 3) and hit["tot"]>50]
    # -------------------------
    #  Setup figura 3D
    # -------------------------
    fig = plt.figure(figsize=(16,9))
    ax = fig.add_subplot(111, projection='3d')

    # -------------------------
    #  Parametri cilindri
    # -------------------------
    L = 15

    # C1
    N1_C1, N2_C1 = 45, 49
    R_C1 = 1.7
    DELTA_ROT = np.deg2rad(10)
    PHI_OFFSET_L1_C1 = 2.609122003 + DELTA_ROT
    PHI_OFFSET_L2_C1 = 3.351032122 + DELTA_ROT
    #bundles_green_C1 = [37, 54]

    # C2
    N1_C2, N2_C2 = 59, 60
    R_C2 = 2.0
    PHI_OFFSET_L1_C2 = 1.5
    PHI_OFFSET_L2_C2 = 2.0
    #bundles_green_C2 = [40, 80]



    # -------------------------
    #  Aspetto grafico
    # -------------------------
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_xlim(-R_C2-1, R_C2+1)
    ax.set_ylim(-R_C2-1, R_C2+1)
    ax.set_zlim(-L, L)
    ax.view_init(elev=7, azim=11)
    ax.text(0, 0, -L-2, "US", color='k', fontsize=16)
    ax.text(0, 0, +L+1, "DS - Readout", color='k', fontsize=16)
    plt.tight_layout()
     #-------------------------
     #Disegna cilindri
     #-------------------------
    plot_cylinder(ax, N1_C1, N2_C1, L, R_C1, PHI_OFFSET_L1_C1, PHI_OFFSET_L2_C1, bundles_green_C1, label="C1")
    plot_cylinder(ax, N1_C2, N2_C2, L, R_C2, PHI_OFFSET_L1_C2, PHI_OFFSET_L2_C2, bundles_green_C2, label="C2", colorin = "teal", colorout = "orange")
    plt.show()


#for i in range(8):
#    mapper_plot(bundles_fired[i])


#for i in range(len(BD_filtered)):
#    if len(BD_filtered[i]) == 4:
#        print(id_filtered[i])
#        mapper_plot(BD_filtered[i])


plt.figure("Bundles", (11,6))
plt.title("Bundles Hit Distribution - Cosmics Rays Acquisition")
plt.hist(bundles_fired_b0, bins = 48, range = (0,64), alpha = 0.5, edgecolor='teal', label = "Board 0")
plt.hist(bundles_fired_b1, bins = 96, range = (0,96), alpha = 0.5, edgecolor='orange', label = "Board 1")
plt.hist(np.array(bundles_fired_b2) + 96, bins = 160, range = (0,160), alpha = 0.5, edgecolor='red', label = "Board 2")
plt.hist(np.array(bundles_fired_b3) + 96, bins = 224, range = (0,224), alpha = 0.5, edgecolor='blue', label = "Board 3")
plt.xlabel("Bundle ID")
plt.ylabel("Number of Hits")
plt.legend()


plt.figure("ToA", (11,6))
plt.title("ToA Distribution - Cosmic Rays Acquisition - 4 bundles, 2 per layer")
counts, bins, patches =  plt.hist(toa_fired, bins = 1000, range = (-0.5,999.5), alpha = 0.5, edgecolor='teal', label = "Time of Arrival")
plt.xlabel("ToA / LSB (0.5 ns)")
plt.ylabel("Number of Hits")


plt.show()
plt.legend()

##Debug
##======================================================
#plt.figure("ToA - Board 0", (11,6))
#plt.title("ToA Distribution - Board 0 - Cosmic Rays Acquisition")
#plt.hist(toa_board0, bins = 1000, range = (-0.5,999.5), alpha = 0.5, edgecolor='teal', label = "Time of Arrival")
#plt.xlabel("ToA / LSB (0.5 ns)")
#plt.ylabel("Number of Hits")
#plt.legend()
#
#plt.figure("ToA - Board 1", (11,6))
#plt.title("ToA Distribution - Board 1 - Cosmic Rays Acquisition")
#plt.hist(toa_board1, bins = 1000, range = (-0.5,999.5), alpha = 0.5, edgecolor='teal', label = "Time of Arrival")
#plt.xlabel("ToA / LSB (0.5 ns)")
#plt.ylabel("Number of Hits")
#plt.legend()
#
#plt.figure("ToT - Board 1", (11,6))
#plt.title(f"ToT Distribution - Board 1 - Cosmic Rays Acquisition - {toa_min} < ToA < {toa_max}")
#plt.hist(tot_board1, bins = 256, range = (-0.5,255.5), alpha = 0.5, edgecolor='teal', label = "Time over Threshold")
#plt.xlabel("ToT / LSB (0.5 ns)")
#plt.ylabel("Number of Hits")
#plt.legend()
##======================================================


plt.figure("ToT", (11,6))
plt.title("ToT Distribution - Cosmic Rays Acquisition")
plt.hist(tot_fired, bins = 256, range = (-0.5,255.5), alpha = 0.5, edgecolor='teal', label = "Time over Threshold")
plt.xlabel("ToT / LSB (0.5 ns)")
plt.ylabel("Number of Hits")
plt.legend()


plt.figure("ToA vs ToT", (11,6))
plt.hist2d(toa_fired, tot_fired, bins=[1000, 256], range=[[-0.5,999.5], [-0.5,255.5]], cmap='turbo')
plt.colorbar(label='Conteggio')
plt.xlabel("ToA [LSB = 0.5 ns]")
plt.ylabel("ToT [LSB = 0.5 ns]")
plt.title("ToA vs ToT distribution")



#c = ROOT.TCanvas("c", "c", 800, 600)
h = ROOT.TH1F("h", "ToT_Histogram", 256, -0.5, 255.5)
for x in tot_fired:
    h.Fill(x)

#h.Fit("landau", "R", "", 0,250)
#h.Draw()
#c.Draw()
#c.SaveAs("hist_fit.pdf")
plt.show()


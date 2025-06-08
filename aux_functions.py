###############################################################
##  Filename: aux_functions.py
##  Author: Pei-Yu Lin
##  Description:
##    This file includes:
##     1. CAN ID Labels and ID Category
##     2. Raw data decoding functions
##     3. Gait phase detection logic
##     4. Parse CAN data into encoder, IMU and GRF depending on CAN ID
##     5. Plotting functions for encoder, IMU, and GRF data
###############################################################

import re
import pandas as pd
import matplotlib.pyplot as plt

# =============================================================
# CAN ID Labels and ID group sets
# =============================================================

encoder_id_label = {
    "Right_Knee_Angle": "0010",
    "Right_Hip_Angle": "0012",
    "Right_Ankle_Angle": "0014",    # no signal
    "Left_Knee_Angle": "0020",
    "Left_Hip_Angle": "0022",
    "Left_Ankle_Angle": "0024",     # no signal
}

imu_id_label = {
    "Right_Femur_RPY_(IMU)": "0016",
    "Left_Femur_RPY_(IMU)": "0026"
}

grf_id_label = {
    # Left foot
    "Left_Ankle_GRF_Channel_1": ("0025", "Channel 1"),
    "Left_Ankle_GRF_Channel_2": ("0025", "Channel 2"),
    "Left_Ankle_GRF_Channel_3": ("0025", "Channel 3"),
    "Left_Ankle_GRF_Channel_4": ("0025", "Channel 4"),
    # Right foot
    "Right_Ankle_GRF_Channel_1": ("0015", "Channel 1"),
    "Right_Ankle_GRF_Channel_2": ("0015", "Channel 2"),
    "Right_Ankle_GRF_Channel_3": ("0015", "Channel 3"),
    "Right_Ankle_GRF_Channel_4": ("0015", "Channel 4")
}

# CAN ID group sets
encoder_ids = {"0010", "0012", "0014", "0020", "0022", "0024"}
imu_ids = {"0016", "0026"}
grf_ids = {"0015", "0025"}
stiffness_ids = {"0011", "0013", "0021", "0023"}

# =============================================================
# 2. Raw Data Decoding Functions
# =============================================================

def decode_phi(data_high, data_low):
    iValue = (data_high << 8) | data_low
    phi = iValue / 65536.0 * 720.0
    return phi

def decode_imu(data):
    x_val = data[0] | (data[1] << 8)
    y_val = data[2] | (data[3] << 8)
    z_val = data[4] | (data[5] << 8)
    return {
        'roll': x_val * 360 / 65535,
        'pitch': (y_val * 360 / 65535) - 180,
        'yaw': (z_val * 360 / 65535) - 180
    }

def decode_grf(data_high, data_low):
    iValue = (data_high << 8) | data_low
    grf = iValue / 65536.0 * 1000.0
    return grf

# =============================================================
# 3. Gait Phase Detection
# =============================================================

def detect_gait_phase(knee, hip, ankle, grf):
    if grf > 20:  # Stance Phase (60% of Gait Cycle)
        if knee < 10 and ankle <= 5 and hip >= 20:
            return 'Initial Contact'
        elif knee >= 10 and ankle < 10 and hip > 15:
            return 'Loading Response'
        elif knee < 5 and 5 <= ankle <= 10 and -5 <= hip <= 5:
            return 'Mid Stance'
        elif ankle > 10 and hip < 0:
            return 'Terminal Stance'
        else:
            return 'Pre-Swing'
    else:  # Swing Phase (40% of Gait Cycle)
        if hip > 15 and knee > 50:
            return 'Initial Swing'
        elif hip > 20 and 30 <= knee <= 50:
            return 'Mid Swing'
        elif knee <= 30 and 15 <= hip <= 25:
            return 'Terminal Swing'
        else:
            return 'Unknown (Swing)'
        
def detect_gait_phase_row(row):
    return detect_gait_phase(row['knee'], row['hip'], row['ankle'], row['grf'])

# =============================================================
# 4. Parse CAN data
# =============================================================

def parse_can_log(file_path, encoder_data, imu_data, grf_data):
    """
    Parse a .trc CAN log file and fill in encoder_data, imu_data, grf_data.

    Parameters:
        file_path:    str, path to the .trc CAN log file
        encoder_data: defaultdict(list), output for encoder angles
        imu_data:     defaultdict(list), output for IMU orientation
        grf_data:     defaultdict(dict(list)), output for GRF channels
    """
        
    pattern = re.compile(
        r"\s*\d+\)\s+([\d.]+)\s+Rx\s+([0-9a-fA-F]{4})\s+\d+\s+((?:[0-9a-fA-F]{2}\s+){1,8})"
    )


    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            match = pattern.match(line)
            if not match:
                continue

            time_offset = float(match.group(1))
            msg_id = match.group(2).upper().zfill(4)
            data_bytes = match.group(3).strip().split()
            data = [int(b, 16) for b in data_bytes]

            # Encoder (5 bytes)
            if msg_id in encoder_ids and len(data) == 5:
                phi = decode_phi(data[3], data[2])
                encoder_data[msg_id].append({
                    "TimeOffset": time_offset,
                    "Angle": phi
                })

            # IMU (6 bytes)
            elif msg_id in imu_ids and len(data) == 6:
                result = decode_imu(data)
                imu_data[msg_id].append({
                    "TimeOffset": time_offset,
                    "Roll (X)": result['roll'],
                    "Pitch (Y)": result['pitch'],
                    "Yaw (Z)": result['yaw']
                })

            # GRF (8 bytes = 4 channels)
            elif msg_id in grf_ids and len(data) == 8:
                for ch in range(4):
                    grf = decode_grf(data[ch * 2 + 1], data[ch * 2])
                    grf_data[msg_id][f"Channel {ch + 1}"].append({
                        "TimeOffset": time_offset,
                        "Value": grf
                    })


# =============================================================
# 5. Plotting Functions
# =============================================================

def plot_encoder(data, name):
    if name not in encoder_id_label:
        print(f"[Error] Unknown encoder name: {name}")
        return
    can_id = encoder_id_label[name]
    if can_id not in data or not data[can_id]:
        print(f"[Warning] No data for {name} (CAN ID {can_id})")
        return
    df = pd.DataFrame(data[can_id])
    plt.figure()
    plt.plot(df["TimeOffset"], df["Angle"], marker='o')
    plt.title(f"{name} - ID {can_id}")
    plt.xlabel("Time Offset (ms)")
    plt.ylabel("Angle (degrees)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_imu(data, name):
    if name not in imu_id_label:
        print(f"[Error] Unknown IMU name: {name}")
        return
    can_id = imu_id_label[name]
    if can_id not in data or not data[can_id]:
        print(f"[Warning] No data for {name} (CAN ID {can_id})")
        return
    df = pd.DataFrame(data[can_id])
    for axis in ["Roll (X)", "Pitch (Y)", "Yaw (Z)"]:
        if axis not in df.columns:
            print(f"[Warning] Missing {axis} in data for {name}")
            continue
        plt.figure()
        plt.plot(df["TimeOffset"], df[axis], label=axis)
        plt.title(f"{name} - {axis} - CAN ID {can_id}")
        plt.xlabel("Time Offset (ms)")
        plt.ylabel("Angle (degrees)")
        plt.grid(True)
        plt.tight_layout()
    plt.show()

def plot_grf(data, name):
    if name not in grf_id_label:
        print(f"[Error] Unknown GRF name: {name}")
        return
    can_id, channel = grf_id_label[name]
    if can_id not in data or channel not in data[can_id] or not data[can_id][channel]:
        print(f"[Warning] No data for {name} (CAN ID {can_id}, {channel})")
        return
    df = pd.DataFrame(data[can_id][channel])
    plt.figure()
    plt.plot(df["TimeOffset"], df["Value"], marker='.')
    plt.title(f"{name} - ID {can_id} / {channel}")
    plt.xlabel("Time Offset (ms)")
    plt.ylabel("Force Value (raw ADC)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
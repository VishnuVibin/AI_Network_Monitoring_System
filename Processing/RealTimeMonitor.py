import subprocess
import csv
import os
from datetime import datetime
from collections import defaultdict

import pandas as pd
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "../data/processed/network_anomaly_model_v3.pkl"

INTERFACE = r"\Device\NPF_Loopback"

WINDOW_SECONDS = 10

LOG_DIR = "../data/logs"
LOG_FILE = os.path.join(LOG_DIR, "network_monitor_log.csv")


# ============================================================
# CREATE LOG DIRECTORY
# ============================================================

os.makedirs(LOG_DIR, exist_ok=True)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading ML model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")
print()


# ============================================================
# FEATURE ORDER
# ============================================================

FEATURE_COLUMNS = [
    "packet_count",
    "total_bytes",
    "average_packet_size",
    "max_packet_size",
    "min_packet_size",
    "unique_source_ips",
    "unique_destination_ips",
    "unique_source_ports",
    "unique_destination_ports",
    "tcp_count",
    "udp_count",
    "icmp_count",
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count"
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_int(value):
    try:
        if value is None or value == "":
            return 0
        return int(float(value))
    except:
        return 0


def safe_float(value):
    try:
        if value is None or value == "":
            return 0.0
        return float(value)
    except:
        return 0.0


def flag_to_int(value):

    if value is None:
        return 0

    value = str(value).strip().lower()

    if value in ["true", "1", "yes"]:
        return 1

    return 0


# ============================================================
# LOG INITIALIZATION
# ============================================================

LOG_COLUMNS = [
    "timestamp",
    "packet_count",
    "total_bytes",
    "average_packet_size",
    "max_packet_size",
    "min_packet_size",
    "unique_source_ips",
    "unique_destination_ips",
    "unique_source_ports",
    "unique_destination_ports",
    "tcp_count",
    "udp_count",
    "icmp_count",
    "syn_count",
    "ack_count",
    "fin_count",
    "rst_count",
    "normal_probability",
    "anomaly_probability",
    "status"
]


if not os.path.exists(LOG_FILE):

    with open(LOG_FILE, "w", newline="") as file:

        writer = csv.writer(file)

        writer.writerow(LOG_COLUMNS)


# ============================================================
# START TSHARK
# ============================================================

print("Starting TShark...")
print("Interface:", INTERFACE)
print()

command = [
    "tshark",
    "-i",
    INTERFACE,
    "-T",
    "fields",

    "-e", "frame.time_epoch",
    "-e", "ip.src",
    "-e", "ip.dst",

    "-e", "tcp.srcport",
    "-e", "tcp.dstport",

    "-e", "udp.srcport",
    "-e", "udp.dstport",

    "-e", "icmp.type",

    "-e", "frame.len",

    "-e", "tcp.flags.syn",
    "-e", "tcp.flags.ack",
    "-e", "tcp.flags.fin",
    "-e", "tcp.flags.reset",

    "-E", "separator=|",
    "-E", "quote=n",
    "-l"
]


process = subprocess.Popen(
    command,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
    bufsize=1
)


# ============================================================
# VARIABLES
# ============================================================

packets = []

window_start = None

first_packet = True


# ============================================================
# PROCESS PACKETS
# ============================================================

try:

    for line in process.stdout:

        line = line.strip()

        if not line:
            continue

        fields = line.split("|")

        if first_packet:

            print("First packet received from TShark!")
            print("Number of fields:", len(fields))
            print("Packet data:")
            print(line)
            print()

            first_packet = False

        if len(fields) < 13:

            print("Invalid packet:", line)

            continue


        # ----------------------------------------------------
        # EXTRACT FIELDS
        # ----------------------------------------------------

        timestamp = safe_float(fields[0])

        source_ip = fields[1]

        destination_ip = fields[2]

        tcp_src_port = fields[3]

        tcp_dst_port = fields[4]

        udp_src_port = fields[5]

        udp_dst_port = fields[6]

        icmp_type = fields[7]

        packet_length = safe_int(fields[8])

        tcp_syn = flag_to_int(fields[9])

        tcp_ack = flag_to_int(fields[10])

        tcp_fin = flag_to_int(fields[11])

        tcp_rst = flag_to_int(fields[12])


        # ----------------------------------------------------
        # START WINDOW
        # ----------------------------------------------------

        if window_start is None:

            window_start = timestamp


        # ----------------------------------------------------
        # STORE PACKET
        # ----------------------------------------------------

        packet = {

            "timestamp": timestamp,

            "source_ip": source_ip,

            "destination_ip": destination_ip,

            "tcp_src_port": tcp_src_port,

            "tcp_dst_port": tcp_dst_port,

            "udp_src_port": udp_src_port,

            "udp_dst_port": udp_dst_port,

            "icmp_type": icmp_type,

            "packet_length": packet_length,

            "tcp_syn": tcp_syn,

            "tcp_ack": tcp_ack,

            "tcp_fin": tcp_fin,

            "tcp_rst": tcp_rst
        }


        packets.append(packet)


        # ----------------------------------------------------
        # CHECK 10 SECOND WINDOW
        # ----------------------------------------------------

        if timestamp - window_start >= WINDOW_SECONDS:

            if len(packets) > 0:

                # ====================================================
                # FEATURE EXTRACTION
                # ====================================================

                packet_count = len(packets)

                total_bytes = sum(
                    p["packet_length"]
                    for p in packets
                )

                packet_sizes = [
                    p["packet_length"]
                    for p in packets
                ]

                average_packet_size = (
                    total_bytes / packet_count
                )

                max_packet_size = max(packet_sizes)

                min_packet_size = min(packet_sizes)


                source_ips = set(
                    p["source_ip"]
                    for p in packets
                    if p["source_ip"]
                )

                destination_ips = set(
                    p["destination_ip"]
                    for p in packets
                    if p["destination_ip"]
                )


                source_ports = set()

                destination_ports = set()


                for p in packets:

                    if p["tcp_src_port"]:
                        source_ports.add(
                            p["tcp_src_port"]
                        )

                    if p["udp_src_port"]:
                        source_ports.add(
                            p["udp_src_port"]
                        )

                    if p["tcp_dst_port"]:
                        destination_ports.add(
                            p["tcp_dst_port"]
                        )

                    if p["udp_dst_port"]:
                        destination_ports.add(
                            p["udp_dst_port"]
                        )


                tcp_count = sum(
                    1
                    for p in packets
                    if p["tcp_src_port"]
                    or p["tcp_dst_port"]
                )


                udp_count = sum(
                    1
                    for p in packets
                    if p["udp_src_port"]
                    or p["udp_dst_port"]
                )


                icmp_count = sum(
                    1
                    for p in packets
                    if p["icmp_type"]
                )


                syn_count = sum(
                    p["tcp_syn"]
                    for p in packets
                )

                ack_count = sum(
                    p["tcp_ack"]
                    for p in packets
                )

                fin_count = sum(
                    p["tcp_fin"]
                    for p in packets
                )

                rst_count = sum(
                    p["tcp_rst"]
                    for p in packets
                )


                # ====================================================
                # CREATE FEATURE DICTIONARY
                # ====================================================

                features = {

                    "packet_count": packet_count,

                    "total_bytes": total_bytes,

                    "average_packet_size":
                        average_packet_size,

                    "max_packet_size":
                        max_packet_size,

                    "min_packet_size":
                        min_packet_size,

                    "unique_source_ips":
                        len(source_ips),

                    "unique_destination_ips":
                        len(destination_ips),

                    "unique_source_ports":
                        len(source_ports),

                    "unique_destination_ports":
                        len(destination_ports),

                    "tcp_count":
                        tcp_count,

                    "udp_count":
                        udp_count,

                    "icmp_count":
                        icmp_count,

                    "syn_count":
                        syn_count,

                    "ack_count":
                        ack_count,

                    "fin_count":
                        fin_count,

                    "rst_count":
                        rst_count
                }


                # ====================================================
                # MODEL PREDICTION
                # ====================================================

                X = pd.DataFrame(
                    [features],
                    columns=FEATURE_COLUMNS
                )


                prediction = model.predict(X)[0]

                probabilities = model.predict_proba(X)[0]


                normal_probability = probabilities[0]

                anomaly_probability = probabilities[1]


                # ====================================================
                # STATUS
                # ====================================================

                if prediction == 1:

                    status = "ANOMALY"

                else:

                    status = "NORMAL"


                # ====================================================
                # DISPLAY RESULT
                # ====================================================

                print("=" * 60)

                print(
                    "[10 SECOND WINDOW]"
                )

                print(
                    "Time:",
                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                )

                print(
                    "Packets captured:",
                    packet_count
                )

                print(
                    "Packet Count:",
                    packet_count
                )

                print(
                    "Total Bytes:",
                    total_bytes
                )

                print(
                    "Average Packet:",
                    round(
                        average_packet_size,
                        2
                    )
                )

                print(
                    "TCP Packets:",
                    tcp_count
                )

                print(
                    "UDP Packets:",
                    udp_count
                )

                print(
                    "ICMP Packets:",
                    icmp_count
                )

                print(
                    "SYN:",
                    syn_count,
                    "ACK:",
                    ack_count,
                    "FIN:",
                    fin_count,
                    "RST:",
                    rst_count
                )


                print()

                print("MODEL RESULT")


                if status == "ANOMALY":

                    print(
                        "🚨 ANOMALY DETECTED!"
                    )

                else:

                    print(
                        "✓ NORMAL TRAFFIC"
                    )


                print(
                    "Normal Probability:",
                    f"{normal_probability * 100:.2f}%"
                )

                print(
                    "Anomaly Probability:",
                    f"{anomaly_probability * 100:.2f}%"
                )


                # ====================================================
                # SAVE TO CSV
                # ====================================================

                log_row = [

                    datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),

                    packet_count,

                    total_bytes,

                    round(
                        average_packet_size,
                        2
                    ),

                    max_packet_size,

                    min_packet_size,

                    len(source_ips),

                    len(destination_ips),

                    len(source_ports),

                    len(destination_ports),

                    tcp_count,

                    udp_count,

                    icmp_count,

                    syn_count,

                    ack_count,

                    fin_count,

                    rst_count,

                    round(
                        normal_probability,
                        4
                    ),

                    round(
                        anomaly_probability,
                        4
                    ),

                    status
                ]


                with open(
                    LOG_FILE,
                    "a",
                    newline=""
                ) as file:

                    writer = csv.writer(file)

                    writer.writerow(log_row)


                print()

                print(
                    "Log saved:",
                    LOG_FILE
                )

                print("=" * 60)

                print()


            # ----------------------------------------------------
            # RESET WINDOW
            # ----------------------------------------------------

            packets = []

            window_start = timestamp


except KeyboardInterrupt:

    print()
    print("Stopping network monitor...")


finally:

    process.terminate()

    print()
    print("Network monitor stopped.")
import subprocess
import pandas as pd
import os
from io import StringIO

# --------------------------------------------------
# File paths
# --------------------------------------------------

PCAP_FILE = "../data/raw/anomaly_loopback_10min.pcapng"
OUTPUT_FILE = "../data/processed/anomaly_loopback_10min_packet_features.csv"

# --------------------------------------------------
# Determine protocol
# --------------------------------------------------

def determine_protocol(row):

    # TCP
    if pd.notna(row["tcp_source_port"]) and row["tcp_source_port"] != "":
        return "TCP"

    # UDP
    if pd.notna(row["udp_source_port"]) and row["udp_source_port"] != "":
        return "UDP"

    # ICMP
    if pd.notna(row["icmp_type"]) and row["icmp_type"] != "":
        return "ICMP"

    # Other protocols
    return "OTHER"


# --------------------------------------------------
# Extract packet features
# --------------------------------------------------

def extract_packets():

    print("Reading PCAP file...")

    command = [
        "tshark",
        "-r", PCAP_FILE,

        "-T", "fields",

        # Timestamp
        "-e", "frame.time_epoch",

        # IP information
        "-e", "ip.src",
        "-e", "ip.dst",

        # TCP
        "-e", "tcp.srcport",
        "-e", "tcp.dstport",

        # UDP
        "-e", "udp.srcport",
        "-e", "udp.dstport",

        # ICMP
        "-e", "icmp.type",

        # Packet size
        "-e", "frame.len",

        # TCP flags
        "-e", "tcp.flags.syn",
        "-e", "tcp.flags.ack",
        "-e", "tcp.flags.fin",
        "-e", "tcp.flags.reset",

        # Output settings
        "-E", "header=y",
        "-E", "separator=\t",
        "-E", "quote=d"
    ]

    # --------------------------------------------------
    # Run TShark
    # --------------------------------------------------

    result = subprocess.run(
        command,
        capture_output=True,
        text=True
    )

    if result.returncode != 0:

        print("\nTShark error:")
        print(result.stderr)

        return

    if not result.stdout.strip():

        print("\nNo packet data found!")

        return

    # --------------------------------------------------
    # Read TShark output
    # --------------------------------------------------

    df = pd.read_csv(
        StringIO(result.stdout),
        sep="\t"
    )

    print(f"Raw packets extracted: {len(df)}")

    # --------------------------------------------------
    # Expected columns
    # --------------------------------------------------

    expected_columns = [

        "frame.time_epoch",

        "ip.src",
        "ip.dst",

        "tcp.srcport",
        "tcp.dstport",

        "udp.srcport",
        "udp.dstport",

        "icmp.type",

        "frame.len",

        "tcp.flags.syn",
        "tcp.flags.ack",
        "tcp.flags.fin",
        "tcp.flags.reset"
    ]

    # Add missing columns
    for column in expected_columns:

        if column not in df.columns:

            df[column] = None

    # --------------------------------------------------
    # Rename columns
    # --------------------------------------------------

    df = df.rename(
        columns={

            "frame.time_epoch":
                "timestamp",

            "ip.src":
                "source_ip",

            "ip.dst":
                "destination_ip",

            "tcp.srcport":
                "tcp_source_port",

            "tcp.dstport":
                "tcp_destination_port",

            "udp.srcport":
                "udp_source_port",

            "udp.dstport":
                "udp_destination_port",

            "icmp.type":
                "icmp_type",

            "frame.len":
                "packet_length",

            "tcp.flags.syn":
                "tcp_syn",

            "tcp.flags.ack":
                "tcp_ack",

            "tcp.flags.fin":
                "tcp_fin",

            "tcp.flags.reset":
                "tcp_rst"
        }
    )

    # --------------------------------------------------
    # Determine protocol
    # --------------------------------------------------

    df["protocol"] = df.apply(
        determine_protocol,
        axis=1
    )

    # --------------------------------------------------
    # Combine TCP and UDP ports
    # --------------------------------------------------

    df["source_port"] = (
        df["tcp_source_port"]
        .fillna(df["udp_source_port"])
    )

    df["destination_port"] = (
        df["tcp_destination_port"]
        .fillna(df["udp_destination_port"])
    )

    # --------------------------------------------------
    # Convert numeric columns
    # --------------------------------------------------

    numeric_columns = [

        "timestamp",

        "source_port",
        "destination_port",

        "packet_length",

        "tcp_syn",
        "tcp_ack",
        "tcp_fin",
        "tcp_rst"
    ]

    for column in numeric_columns:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0)

    # --------------------------------------------------
    # Remove packets without IP information
    # --------------------------------------------------

    df = df[
        df["source_ip"].notna()
        &
        df["destination_ip"].notna()
    ]

    # --------------------------------------------------
    # Select final features
    # --------------------------------------------------

    df = df[

        [

            "timestamp",

            "source_ip",
            "destination_ip",

            "source_port",
            "destination_port",

            "protocol",

            "packet_length",

            "tcp_syn",
            "tcp_ack",
            "tcp_fin",
            "tcp_rst"

        ]
    ]

    # --------------------------------------------------
    # Create output directory
    # --------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    # --------------------------------------------------
    # Display results
    # --------------------------------------------------

    print("\nFeature extraction completed!")

    print(
        f"Packets extracted: {len(df)}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print("\nProtocol distribution:")

    print(
        df["protocol"].value_counts()
    )

    print("\nFirst 5 packets:")

    print(
        df.head()
    )


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    extract_packets()
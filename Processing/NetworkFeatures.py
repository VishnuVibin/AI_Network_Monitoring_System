import pandas as pd
import os

INPUT_FILE = "../data/processed/anomaly_loopback_10min_packet_features.csv"
OUTPUT_FILE = "../data/processed/anomaly_loopback_10min_features.csv"
WINDOW_SECONDS = 10


def create_network_features():

    print("Loading packet features...")

    df = pd.read_csv(INPUT_FILE)

    if df.empty:
        print("No packet data found!")
        return

    print(f"Packets loaded: {len(df)}")

    # --------------------------------------------------
    # Convert timestamp
    # --------------------------------------------------

    df["timestamp"] = pd.to_numeric(
        df["timestamp"],
        errors="coerce"
    )

    df = df.dropna(subset=["timestamp"])

    # --------------------------------------------------
    # Create relative time
    # --------------------------------------------------

    start_time = df["timestamp"].min()

    df["relative_time"] = (
        df["timestamp"] - start_time
    )

    # Create 10-second window
    df["window"] = (
        df["relative_time"] // WINDOW_SECONDS
    ).astype(int)

    # --------------------------------------------------
    # Create protocol helper columns
    # --------------------------------------------------

    df["is_tcp"] = (
        df["protocol"].str.upper() == "TCP"
    ).astype(int)

    df["is_udp"] = (
        df["protocol"].str.upper() == "UDP"
    ).astype(int)

    df["is_icmp"] = (
        df["protocol"].str.upper() == "ICMP"
    ).astype(int)

    # --------------------------------------------------
    # Aggregate network features
    # --------------------------------------------------

    grouped = df.groupby("window")

    features = grouped.agg(

        packet_count=(
            "packet_length",
            "count"
        ),

        total_bytes=(
            "packet_length",
            "sum"
        ),

        average_packet_size=(
            "packet_length",
            "mean"
        ),

        max_packet_size=(
            "packet_length",
            "max"
        ),

        min_packet_size=(
            "packet_length",
            "min"
        ),

        unique_source_ips=(
            "source_ip",
            "nunique"
        ),

        unique_destination_ips=(
            "destination_ip",
            "nunique"
        ),

        unique_source_ports=(
            "source_port",
            "nunique"
        ),

        unique_destination_ports=(
            "destination_port",
            "nunique"
        ),

        tcp_count=(
            "is_tcp",
            "sum"
        ),

        udp_count=(
            "is_udp",
            "sum"
        ),

        icmp_count=(
            "is_icmp",
            "sum"
        ),

        syn_count=(
            "tcp_syn",
            "sum"
        ),

        ack_count=(
            "tcp_ack",
            "sum"
        ),

        fin_count=(
            "tcp_fin",
            "sum"
        ),

        rst_count=(
            "tcp_rst",
            "sum"
        )
    ).reset_index()

    # --------------------------------------------------
    # Add actual time window
    # --------------------------------------------------

    grouped_time = grouped["timestamp"].min()

    features["timestamp"] = (
        grouped_time
        .reset_index(drop=True)
    )

    # --------------------------------------------------
    # Reorder columns
    # --------------------------------------------------

    columns = [
        "timestamp",
        "window",
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

    features = features[columns]

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    os.makedirs(
        os.path.dirname(OUTPUT_FILE),
        exist_ok=True
    )

    features.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\nNetwork feature extraction completed!")

    print(
        f"10-second windows created: {len(features)}"
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print("\nNetwork features:")
    print(features)


if __name__ == "__main__":
    create_network_features()
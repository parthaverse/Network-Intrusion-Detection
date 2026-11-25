
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create folders
os.makedirs("graphs", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Dataset column names
columns = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land",
    "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
    "attack_type", "difficulty"
]

# Load dataset
df = pd.read_csv(
    "dataset/KDDTrain+_20Percent.txt",
    header=None,
    names=columns
)

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 records:")
print(df.head())

print("\nMissing values:", df.isnull().sum().sum())
print("Duplicate records:", df.duplicated().sum())

# Data cleaning
df = df.dropna()
df = df.drop_duplicates()

# Binary classification
# Normal = 0, Attack = 1
df["label"] = df["attack_type"].apply(
    lambda x: 0 if x == "normal" else 1
)

df["traffic_type"] = df["label"].map({
    0: "Normal",
    1: "Malicious"
})

print("\nTraffic distribution:")
print(df["traffic_type"].value_counts())

print("\nAttack types:")
print(df["attack_type"].value_counts())

sns.set_theme(style="whitegrid")

# Graph 1: Normal vs malicious traffic
plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="traffic_type",
    hue="traffic_type",
    palette="Set2"
)
plt.title("Normal vs Malicious Traffic")
plt.xlabel("Traffic Type")
plt.ylabel("Number of Records")
plt.tight_layout()
plt.savefig("graphs/class_distribution.png", dpi=150)
plt.close()

# Graph 2: Traffic percentage
plt.figure(figsize=(7, 7))
df["traffic_type"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%",
    colors=["skyblue", "salmon"]
)
plt.title("Network Traffic Percentage")
plt.ylabel("")
plt.tight_layout()
plt.savefig("graphs/traffic_percentage.png", dpi=150)
plt.close()

# Graph 3: Protocol distribution
plt.figure(figsize=(8, 5))
sns.countplot(
    data=df,
    x="protocol_type",
    color="steelblue"
)
plt.title("Network Protocol Distribution")
plt.tight_layout()
plt.savefig("graphs/protocol_distribution.png", dpi=150)
plt.close()

# Graph 4: Top 10 attack types
plt.figure(figsize=(10, 6))
attacks = df[df["label"] == 1]

attacks["attack_type"].value_counts().head(10).plot(
    kind="barh",
    color="tomato"
)
plt.title("Top 10 Network Attack Types")
plt.xlabel("Number of Attacks")
plt.tight_layout()
plt.savefig("graphs/attack_distribution.png", dpi=150)
plt.close()

# Graph 5: Top 10 network services
plt.figure(figsize=(10, 6))
df["service"].value_counts().head(10).plot(
    kind="barh",
    color="teal"
)
plt.title("Top 10 Network Services")
plt.xlabel("Number of Records")
plt.tight_layout()
plt.savefig("graphs/service_distribution.png", dpi=150)
plt.close()

# Graph 6: Protocol-wise traffic
plt.figure(figsize=(9, 5))
sns.countplot(
    data=df,
    x="protocol_type",
    hue="traffic_type"
)
plt.title("Protocol-wise Normal and Malicious Traffic")
plt.tight_layout()
plt.savefig("graphs/protocol_traffic.png", dpi=150)
plt.close()

# Graph 7: Network connection flags
plt.figure(figsize=(10, 5))
sns.countplot(
    data=df,
    x="flag",
    color="mediumpurple"
)
plt.title("Network Connection Flag Distribution")
plt.tight_layout()
plt.savefig("graphs/flag_distribution.png", dpi=150)
plt.close()

# Graph 8: Correlation heatmap
features = [
    "duration", "src_bytes", "dst_bytes",
    "count", "srv_count", "serror_rate",
    "rerror_rate", "same_srv_rate",
    "diff_srv_rate", "label"
]

plt.figure(figsize=(12, 8))
sns.heatmap(
    df[features].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Network Features Correlation")
plt.tight_layout()
plt.savefig("graphs/correlation_heatmap.png", dpi=150)
plt.close()

# Save cleaned dataset
df.to_csv("dataset/cleaned_data.csv", index=False)

# Save statistical summary
df.describe().to_csv("results/dataset_summary.csv")

print("\n==============================")
print("DATA ANALYSIS COMPLETED!")
print("Cleaned dataset saved.")
print("8 graphs generated successfully.")
print("Dataset summary saved.")
print("==============================")

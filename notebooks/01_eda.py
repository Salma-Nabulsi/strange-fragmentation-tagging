import uproot
import awkward as ak
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
tuplename = '/Users/lamanazer/Downloads/DTT_Kmunu_small.root'
treename  = "DecayTree"
outdir    = '/Users/lamanazer/Desktop/frag plots'
os.makedirs(outdir, exist_ok=True)
branches2 = [
    'Tr_AALLSAMEBPV', 'Tr_ACHI2DOCA', 'Tr_ADOCA', 'Tr_BPVIP', 'Tr_BPVIPCHI2', 'Tr_Charge', 'Tr_E', 'Tr_EcalE', 'Tr_Eta', 'Tr_HcalE', 'Tr_MinIP','Tr_MinIPChi2', 'Tr_ORIG_FLAGS', 'Tr_P', 'Tr_PIDK', 'Tr_PIDe', 'Tr_PIDmu','Tr_PIDp', 'Tr_PROBNNe', 'Tr_PROBNNk', 'Tr_PROBNNmu', 'Tr_PROBNNp','Tr_PROBNNpi', 'Tr_PT', 'Tr_PZ', 'Tr_Phi', 'Tr_PrsE', 'Tr_THETA','Tr_TRCHI2DOF', 'Tr_TRFITMATCHCHI2', 'Tr_TRFITTCHI2','Tr_TRFITVELOCHI2NDOF', 'Tr_TRGHOSTPROB', 'Tr_TRTYPE','Tr_TrFIRSTHITZ', 'Tr_TrFITTCHI2NDOF', 'Tr_VeloCharge']
branches3 = branches2 + ["Tr_MC_ID"] #This is useful for when comparing traditional vs strange fragmentation
features  = branches2

#Loading data 
tree = uproot.open(tuplename)[treename]
arr  = tree.arrays(branches3, library="ak")
tracks = ak.zip({k: arr[k] for k in branches3})
flat   = ak.flatten(tracks, axis=1)
df     = pd.DataFrame({k: ak.to_numpy(flat[k]) for k in branches3})

# Flatten all tracks to same length 
lens = {k: len(ak.flatten(arr[k], axis=1)) for k in branches3}
assert len(set(lens.values())) == 1, f"Problem in flattened lengths: {lens}"
print(f"Loaded {len(df):,} tracks")

# Add event & track index cols
ref            = arr[branches3[0]]
evt_idx_jagged = ak.broadcast_arrays(ak.local_index(ref, axis=0), ref)[0]
trk_idx_jagged = ak.local_index(ref, axis=1)
df["event_idx"] = ak.to_numpy(ak.flatten(evt_idx_jagged, axis=1))
df["track_idx"] = ak.to_numpy(ak.flatten(trk_idx_jagged, axis=1))

#Labels 
df['y_gen'] = ((df["Tr_AALLSAMEBPV"] == 0) & (df["Tr_ORIG_FLAGS"]  == 1)).astype(int)

# Trakc level balance (distribution)
print("\n── Class balance ──")
print(df["y_gen"].value_counts())
print(f"Signal fraction : {df['y_gen'].mean():.4f}  (1 in {1/df['y_gen'].mean():.0f} tracks)")
print(f"Signal tracks   : {df['y_gen'].sum():,}")
print(f"Background tracks: {(df['y_gen']==0).sum():,}")
# very imbalanced - roughly 1 signal per X background tracks
#  Event-level balance 
n_events      = df["event_idx"].nunique()
sig_events    = df.groupby("event_idx")["y_gen"].max().sum()
print(f"\nTotal events           : {n_events:,}")
print(f"Events with signal     : {sig_events:,}  ({sig_events/n_events:.2%})")
print(f"Events without signal  : {n_events - sig_events:,}")
print("\nStatistics descriptions for data")
desc = df[features].describe()
print(desc)
desc.to_csv(os.path.join(outdir, "describe.csv"))
print(f"Saved describe.csv to {outdir}")

#Missing values 
print("\nMissing values")
nulls = df[features].isnull().sum()
nulls = nulls[nulls > 0]
if len(nulls) == 0:
    print("No missing values found.")
else:
    print(nulls)

#Zero variance value test
print("\nFeature variance")
low_var = df[features].var().sort_values()
print(low_var.head(10))

#Boxplots
fig, ax = plt.subplots(figsize=(8, 5))
df.boxplot(column=['Tr_ADOCA', 'Tr_BPVIPCHI2', 'Tr_TRGHOSTPROB'], ax=ax)
ax.set_title("Boxplot of key variables")
ax.set_ylabel("Value")
plt.tight_layout()
plt.savefig(os.path.join(outdir, "boxplot.png"), dpi=150)
plt.close()
#Signal vs background
# plot signal vs bkg for each feature - checking separation
plot_dir = os.path.join(outdir, "feature_plots")
os.makedirs(plot_dir, exist_ok=True)
# Tr_PIDK and Tr_PROBNNk expected to be most discriminating for kaon ID
for x in features:
    s1 = df.loc[df.y_gen == 1, x].dropna()
    s0 = df.loc[df.y_gen == 0, x].dropna()

    if not (pd.api.types.is_bool_dtype(df[x]) or pd.api.types.is_numeric_dtype(df[x])):
        continue

    fig, ax = plt.subplots(figsize=(6, 4))

    if s1.dtype == bool:
        bins = [-0.5, 0.5, 1.5]
        ax.hist(s1.astype(int), bins=bins, density=True,histtype="step", linewidth=1.5, label="Signal")
        ax.hist(s0.astype(int), bins=bins, density=True,histtype="step", linewidth=1.5, label="Background")
    else:
        # making sure sig and bkg are same scale 
        combined    = pd.concat([s1, s0])
        bin_edges   = np.linspace(combined.quantile(0.001),
                                  combined.quantile(0.999), 51)
        ax.hist(s1, bins=bin_edges, density=True, histtype="step", linewidth=1.5, label="Signal")
        ax.hist(s0, bins=bin_edges, density=True, histtype="step", linewidth=1.5, label="Background")

    ax.set_xlabel(x, fontsize=11)
    ax.set_ylabel("Normalised:", fontsize=10)
    ax.set_title(f"Signal vs Background{x}", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(plot_dir, f"pair_{x}.png"), dpi=120)
    plt.close(fig)
print(f"Saved {len(features)} feature plots to {plot_dir}")

#looking for correlations
print("\nCorrelation")
key_vars = ['Tr_ADOCA', 'Tr_BPVIPCHI2', 'Tr_TRGHOSTPROB', 'Tr_MinIP','Tr_PIDK', 'Tr_PROBNNk', 'Tr_PT', 'Tr_BPVIP']
print(df[key_vars].corr().round(3))

#Heatmap 
corr = df[features].corr(numeric_only=True)
fig, ax = plt.subplots(figsize=(16, 13))
sns.heatmap( corr, vmin=-1, vmax=1, center=0,cmap="coolwarm",linewidths=0.3,square=False,ax=ax,cbar_kws={"shrink": 0.7})
ax.set_title("Full feature correlation matrix", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(outdir, "correlation_heatmap.png"), dpi=150)
plt.close()

#Clustermap
cg = sns.clustermap(corr,cmap="coolwarm",center=0,figsize=(16, 14),dendrogram_ratio=0.1, cbar_pos=(0.02, 0.8, 0.03, 0.15))
cg.fig.suptitle("Clustered correlation matrix", y=1.01, fontsize=13)
cg.savefig(os.path.join(outdir, "correlation_clustermap.png"), dpi=150)
plt.close()

#correlated pairs
print("\ncorrelated pairs")
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
high_pairs = [
    (c1, c2, corr.loc[c1, c2])
    for c1 in upper.columns
    for c2 in upper.columns
    if c1 < c2 and abs(upper.loc[c1, c2]) > 0.90
]
if high_pairs:
    for c1, c2, v in sorted(high_pairs, key=lambda x: -abs(x[2])):
        print(f"  {c1} &  {c2}  :  {v:.3f}")
else:
    print("  No pairs overr 0.90")

print(f"\n EDA finiahsed. All outputs are saved")
os.system(f'open "{outdir}"')

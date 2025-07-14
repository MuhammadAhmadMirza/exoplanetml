import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Optional: use dark theme
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# Load the file using the actual format
df = pd.read_csv("./AstroImaging/Raw Data.tbl", delim_whitespace=True, header=None, dtype=str)

# Rename only the relevant columns based on column indices
df = df.rename(columns={
    0: 'Filename',
    3: 'Time_BJD',
    4: 'Time_JD',
    11: 'AIRMASS',
    13: 'MAG_AUTO',
    14: 'MAGERR_AUTO',
    15: 'EXPTIME',
    16: 'FWHM',
    17: 'FLUX_AUTO',
    18: 'FLUXERR_AUTO',
    19: 'BACKGROUND'
})

# Convert relevant columns to numeric, coercing errors to NaN
numeric_cols = ['Time_BJD', 'MAG_AUTO', 'MAGERR_AUTO', 'FLUX_AUTO', 'FLUXERR_AUTO', 'AIRMASS', 'FWHM', 'EXPTIME', 'BACKGROUND']
df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

# Drop rows with any missing values in relevant columns
df_clean = df.dropna(subset=numeric_cols)

# ---- Plot 1: Light Curve ----
plt.figure(figsize=(10, 5))
plt.scatter(df_clean['Time_BJD'], df_clean['MAG_AUTO'], s=10, alpha=0.7)
plt.gca().invert_yaxis()
plt.title("Light Curve (Magnitude vs Time)")
plt.xlabel("Time (BJD)")
plt.ylabel("Magnitude")
plt.tight_layout()
plt.savefig("./AstroImaging/plot1_light_curve.png")
plt.close()

# ---- Plot 2: Flux vs Time ----
plt.figure(figsize=(10, 5))
plt.plot(df_clean['Time_BJD'], df_clean['FLUX_AUTO'], lw=0.7)
plt.title("Flux over Time")
plt.xlabel("Time (BJD)")
plt.ylabel("Flux")
plt.tight_layout()
plt.savefig("./AstroImaging/plot2_flux_time.png")
plt.close()

# ---- Plot 3: Flux vs Magnitude ----
sns.jointplot(data=df_clean, x='MAG_AUTO', y='FLUX_AUTO', kind='hex', height=6)
plt.suptitle("Flux vs Magnitude", y=1.02)
plt.savefig("./AstroImaging/plot3_flux_vs_mag.png")
plt.close()

# ---- Plot 4: Airmass vs Magnitude ----
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_clean, x='AIRMASS', y='MAG_AUTO')
plt.gca().invert_yaxis()
plt.title("Airmass vs Magnitude")
plt.tight_layout()
plt.savefig("./AstroImaging/plot4_airmass_mag.png")
plt.close()

# ---- Plot 5: Histogram of Exposure Times ----
plt.figure(figsize=(7, 4))
sns.histplot(df_clean['EXPTIME'], bins=20, kde=True)
plt.title("Exposure Time Distribution")
plt.xlabel("Exposure Time (s)")
plt.tight_layout()
plt.savefig("./AstroImaging/plot5_exptime_hist.png")
plt.close()

# ---- Plot 6: FWHM vs Magnitude ----
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_clean, x='FWHM', y='MAG_AUTO')
plt.gca().invert_yaxis()
plt.title("FWHM vs Magnitude (Seeing Conditions)")
plt.tight_layout()
plt.savefig("./AstroImaging/plot6_fwhm_mag.png")
plt.close()

# ---- Plot 7: Error in Flux vs Flux ----
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_clean, x='FLUX_AUTO', y='FLUXERR_AUTO', alpha=0.5)
plt.title("Flux Error vs Flux")
plt.xlabel("Flux")
plt.ylabel("Flux Error")
plt.tight_layout()
plt.savefig("./AstroImaging/plot7_fluxerr_flux.png")
plt.close()

# ---- Plot 8: Background vs Flux ----
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_clean, x='BACKGROUND', y='FLUX_AUTO')
plt.title("Background Brightness vs Flux")
plt.tight_layout()
plt.savefig("./AstroImaging/plot8_background_flux.png")
plt.close()

# ---- Plot 9: Magnitude Error Distribution ----
plt.figure(figsize=(7, 4))
sns.histplot(df_clean['MAGERR_AUTO'], bins=30, kde=True)
plt.title("Magnitude Error Distribution")
plt.tight_layout()
plt.savefig("./AstroImaging/plot9_magerr_hist.png")
plt.close()

# ---- Plot 10: Correlation Heatmap ----
plt.figure(figsize=(10, 6))
corr = df_clean[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("./AstroImaging/plot10_correlation_heatmap.png")
plt.close()

print("All 10 plots have been saved as PNG files.")

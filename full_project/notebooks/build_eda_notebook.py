"""Build eda_phase0.ipynb from the EDA script."""
import nbformat, textwrap

nb = nbformat.v4.new_notebook()
nb.metadata["kernelspec"] = {
    "display_name": "Python 3 (.venv)",
    "language": "python",
    "name": "python3"
}

def code(src): return nbformat.v4.new_code_cell(textwrap.dedent(src))
def md(src):   return nbformat.v4.new_markdown_cell(textwrap.dedent(src))

nb.cells = [
    md("# Phase 0 — Exploratory Data Analysis\n\n**Beyond WhAM: Self-Supervised Rhythm-Spectral Alignment for Sperm Whale Coda Understanding**\n\nGoal: understand the DSWP dataset before building any model. Produce the 8 standard EDA figures."),
    code("""\
        import os, warnings
        warnings.filterwarnings("ignore")
        import numpy as np
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns
        import librosa
        from sklearn.manifold import TSNE
        from sklearn.preprocessing import StandardScaler

        BASE   = "/Users/joaoquintanilha/Downloads/data_297_final_paper/full_project"
        DATA   = f"{BASE}/datasets"
        AUDIO  = f"{DATA}/dswp_audio"
        FIGDIR = f"{BASE}/figures/phase0"
        os.makedirs(FIGDIR, exist_ok=True)

        UNIT_COLORS = {"A": "#2196F3", "D": "#FF9800", "F": "#4CAF50"}
        sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

        df = pd.read_csv(f"{DATA}/dswp_labels.csv")
        clean = df[df.is_noise == 0].reset_index(drop=True)
        clean["ici_list"] = clean["ici_sequence"].apply(
            lambda s: [float(x)*1000 for x in str(s).split("|")] if not pd.isna(s) else [])
        clean["mean_ici_ms"] = clean["ici_list"].apply(lambda x: np.mean(x) if x else np.nan)
        clean["year"] = pd.to_datetime(clean["date"], dayfirst=True).dt.year
        print(f"Total={len(df)}, Clean={len(clean)}, Noise={len(df)-len(clean)}")
        print("Unit counts (clean):", dict(clean.unit.value_counts().sort_index()))
        """),
    md("## Fig 1 — Label Distributions"),
    code("""\
        exec(open(f"{BASE}/notebooks/eda_phase0.py").read().split("# FIG 1")[1].split("# FIG 2")[0])
        # (figures saved via the script; display from file)
        from IPython.display import Image
        Image(f"{FIGDIR}/fig1_label_distributions.png")
        """),
    md("## All 8 Figures (from pre-executed script)"),
    code("""\
        import subprocess, sys
        result = subprocess.run([sys.executable,
                                 f"{BASE}/notebooks/eda_phase0.py"],
                                capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print("STDERR:", result.stderr)
        """),
    md("## Display Figures"),
    code("""\
        from IPython.display import Image, display
        for i in range(1, 9):
            fig_files = sorted([f for f in os.listdir(FIGDIR) if f.startswith(f"fig{i}")])
            for f in fig_files:
                print(f)
                display(Image(f"{FIGDIR}/{f}"))
        """),
    md("## Summary Statistics"),
    code("""\
        idn0_mask = clean["individual_id"].astype(str) == "0"
        labeled = clean[~idn0_mask]
        print(f"Total codas: {len(df)}")
        print(f"Clean codas: {len(clean)} (noise: {len(df)-len(clean)})")
        print(f"Social units (clean): {dict(clean.unit.value_counts().sort_index())}")
        print(f"Social units (all):   {dict(df.unit.value_counts().sort_index())}")
        print(f"Unique coda types (clean): {clean.coda_type.nunique()}")
        print(f"IDN=0: {idn0_mask.sum()} ({idn0_mask.mean()*100:.1f}%)")
        print(f"IDN labeled: {len(labeled)}, {labeled.individual_id.nunique()} individuals")
        print(f"Mean duration: {clean.duration_sec.mean():.3f}s")
        print(f"Mean ICI: {clean.mean_ici_ms.mean():.1f}ms (std={clean.mean_ici_ms.std():.1f}ms)")
        print(f"Date range: {clean.year.min()} – {clean.year.max()}")
        """),
]

out_path = f"/Users/joaoquintanilha/Downloads/data_297_final_paper/full_project/notebooks/eda_phase0.ipynb"
with open(out_path, "w") as f:
    nbformat.write(nb, f)
print(f"Notebook written to {out_path}")

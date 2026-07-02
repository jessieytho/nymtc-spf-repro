"""V3: anchor NYMTC recovered fatality IRRs to the national FARS/FHWA
fatality-rate-per-100M-VMT series (NHTSA DOT HS 813 756, Sep 2025 vintage).
Build the overlay figure + the quantitative agreement metrics + series CSV."""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="outputs"
# National series (consistent vintage: 2013-2022 FARS Final, 2023 ARF, 2024 projection; VMT FHWA Jun-2025 TVT)
nat=pd.DataFrame({"Year":[2019,2020,2021,2022,2023,2024],
                  "fatalities":[36355,39007,43230,42721,40901,39345],
                  "rate_per100M":[1.11,1.34,1.38,1.34,1.26,1.20]})
nat["rate_ratio_2019"]=nat.rate_per100M/nat.rate_per100M.iloc[0]
nat.to_csv(f"{OUT}/v3_national_fars_series.csv",index=False)
# NYMTC fatality IRRs (grid-inverted CIs of record)
g=pd.read_csv(f"{OUT}/r4_grid_inverted_ci.csv"); f=g[g.Outcome=="Fatalities"].sort_values("Year")
yrs=[2019]+list(f.Year); irr=[1.0]+list(f.IRR); lo=[1.0]+list(f.grid_CI_lo); hi=[1.0]+list(f.grid_CI_hi)
# agreement metrics (2020-2024)
nymtc=np.array(list(f.IRR)); natr=nat.rate_ratio_2019.values[1:]
r=np.corrcoef(nymtc,natr)[0,1]; gap=nymtc-natr
print("National rate ratio vs 2019:", [round(x,3) for x in nat.rate_ratio_2019])
print("NYMTC fatality IRR vs 2019: ", [round(x,3) for x in nymtc])
print(f"Pearson r (shape agreement) = {r:.3f}")
print(f"NYMTC minus national (IRR points): {[round(x,3) for x in gap]}  mean={gap.mean():.3f}")
print(f"Both peak in {2020+int(np.argmax(nymtc))} (NYMTC) and {2020+int(np.argmax(natr))} (national)")
# overlay figure
fig,ax=plt.subplots(figsize=(7.6,4.7))
ax.fill_between(yrs,lo,hi,color="#c53030",alpha=0.13,zorder=1,label="NYMTC 95% CI (grid-inverted WCR)")
ax.plot(yrs,irr,marker="s",color="#c53030",lw=2.2,zorder=3,label="NYMTC fatality rate ratio (model IRR)")
ax.plot(nat.Year,nat.rate_ratio_2019,marker="o",color="#1f4e79",lw=2.2,ls="--",zorder=3,
        label="National FARS/FHWA fatality rate ratio")
ax.axhline(1.0,color="#666",lw=1,ls=":")
ax.set_xticks(yrs); ax.set_ylabel("Fatality rate ratio vs 2019"); ax.set_xlabel("Year")
ax.set_title("External benchmark: NYMTC's recovered fatality-rate movement tracks the\n"
             "national FARS series (same 2021 peak, same shape) \u2014 regionally amplified and more persistent",
             fontsize=10,weight="bold")
ax.legend(frameon=False,fontsize=8.5,loc="upper right")
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.annotate("national nearly back to baseline by 2024 (1.08);\nNYMTC still elevated (1.20)",
            xy=(2024,1.195),xytext=(2021.2,1.05),fontsize=8,color="#333",
            arrowprops=dict(arrowstyle="->",color="#888",lw=0.8))
fig.tight_layout(); fig.savefig(f"{OUT}/fig_v3_national_benchmark.png",dpi=200,bbox_inches="tight")
print("\nSaved v3_national_fars_series.csv, fig_v3_national_benchmark.png")

"""Regenerate fig_divergence.png with the grid-inverted WCR bands (intervals of
record from R4b), replacing the original asymptotic-band figure."""
import pandas as pd, numpy as np, os
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT="outputs"
ci=pd.read_csv(f"{OUT}/r4_grid_inverted_ci.csv")
years=[2019,2020,2021,2022,2023,2024]
style={"Crashes":("#2b6cb0","o","-"),"Fatalities":("#c53030","s","-"),"Serious injuries":("#2f855a","^","--")}
name_map={"Crashes":"Crashes","Fatalities":"Fatalities","Serious injuries":"Serious injuries"}
fig,ax=plt.subplots(figsize=(7.4,4.6))
for oc,(c,mk,ls) in style.items():
    sub=ci[ci.Outcome==oc].sort_values("Year")
    irr=[1.0]+list(sub.IRR); lo=[1.0]+list(sub.grid_CI_lo); hi=[1.0]+list(sub.grid_CI_hi)
    ax.plot(years,irr,marker=mk,color=c,ls=ls,lw=2,label=oc,zorder=3)
    ax.fill_between(years,lo,hi,color=c,alpha=0.13,zorder=1)
ax.axhline(1.0,color="#555",lw=1,ls=":")
ax.set_ylabel("Rate ratio vs 2019  (per 100M VMT, net of county FE)")
ax.set_xlabel("Year"); ax.set_xticks(years)
ax.set_title("Emptier but deadlier: exposure-normalized NYMTC safety, 2019–2024\n"
             "(bands = grid-inverted wild-cluster-bootstrap 95% CIs, G=10)",
             fontsize=10.5,weight="bold")
ax.legend(frameon=False,fontsize=9,loc="upper left")
for s in ("top","right"): ax.spines[s].set_visible(False)
ax.annotate("crashes flat-to-down",(2024,ci[(ci.Outcome=='Crashes')&(ci.Year==2024)].IRR.iloc[0]),
            xytext=(2021.6,0.74),fontsize=8.5,color="#2b6cb0")
ax.annotate("fatality rate elevated\n(2023 attenuates)",(2021,ci[(ci.Outcome=='Fatalities')&(ci.Year==2021)].IRR.iloc[0]),
            xytext=(2021.1,1.46),fontsize=8.5,color="#c53030")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_divergence.png",dpi=200,bbox_inches="tight")
print("Saved fig_divergence.png with grid-inverted bands")

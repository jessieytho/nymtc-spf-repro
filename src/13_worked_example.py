"""Worked practitioner example: a single county whose raw counts read as a safety
win while its exposure-normalized fatality rate deteriorates. Reads ./data, writes
./outputs/worked_example.csv. Dependency-light (pandas)."""
import os, pandas as pd
os.makedirs("outputs", exist_ok=True)
df = pd.read_csv("data/nymtc_safety_panel.csv")
a = df[df.Year == 2019].set_index("County")
b = df[df.Year == 2024].set_index("County")
rows = []
for c in a.index:
    rows.append({
        "county": c.title(),
        "crashes_2019": int(a.loc[c, "crashes"]), "crashes_2024": int(b.loc[c, "crashes"]),
        "crashes_pct": 100*(b.loc[c,"crashes"]-a.loc[c,"crashes"])/a.loc[c,"crashes"],
        "fatalities_2019": int(a.loc[c,"fatalities"]), "fatalities_2024": int(b.loc[c,"fatalities"]),
        "fatalities_pct": 100*(b.loc[c,"fatalities"]-a.loc[c,"fatalities"])/a.loc[c,"fatalities"],
        "vmt_pct": 100*(b.loc[c,"VMT_100M"]-a.loc[c,"VMT_100M"])/a.loc[c,"VMT_100M"],
        "crash_rate_RR": b.loc[c,"crash_rate_per100M"]/a.loc[c,"crash_rate_per100M"],
        "fatal_rate_RR": b.loc[c,"fatal_rate_per100M"]/a.loc[c,"fatal_rate_per100M"],
    })
out = pd.DataFrame(rows).round(3)
# "flip" = raw counts read as improvement (crashes down, fatalities not up materially)
# while the exposure-normalized fatality rate rises.
out["flip_counts_say_safer_rate_says_worse"] = (out.crashes_pct < 0) & (out.fatalities_pct <= 5) & (out.fatal_rate_RR > 1.05)
out.to_csv("outputs/worked_example.csv", index=False)
q = out[out.county == "Queens"].iloc[0]
print(out.to_string(index=False))
print("\nWORKED EXAMPLE (Queens):")
print(f"  crashes {q.crashes_2019}->{q.crashes_2024} ({q.crashes_pct:+.1f}%); "
      f"fatalities {q.fatalities_2019}->{q.fatalities_2024} ({q.fatalities_pct:+.1f}%); "
      f"VMT {q.vmt_pct:+.1f}%")
print(f"  crash-rate RR {q.crash_rate_RR:.2f}; fatal-rate RR {q.fatal_rate_RR:.2f}")

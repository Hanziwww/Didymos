"""icutwin on the AutoDL host: the authenticity audit's explain measure in compact text (aggregates only):
    python auth_explain.py <explain.json> [top]
per dataset and view, the unigram-only discriminator (twins; positive control), the most different tokens and kind
bigrams (Welch's t; twins/real), tokens per kind and events per time group."""
import json
import sys
from pathlib import Path

e = json.loads(Path(sys.argv[1]).read_text())
top = int(sys.argv[2]) if len(sys.argv) > 2 else 10
for ds, views in e.items():
    for name, x in views.items():
        c = x["positive_control"]
        print(f"== {ds} {name}: unigram AUC {x['unigram_auc']:.3f} (control {c['unigram_auc']:.3f}); "
              f"max |t| control: tokens {c['max_abs_t']['tokens']}, kind bigrams {c['max_abs_t']['kind_bigrams']}")
        for y in x["tokens"][:top]:
            print(f"   {y['t']:+7.1f}  {y['token'][:60]:60s} per stay {y['per_stay'][0]:8.3f} / {y['per_stay'][1]:8.3f}"
                  f"  stays {y['stays'][0]:.2f} / {y['stays'][1]:.2f}")
        print("   kind bigrams:", "; ".join(f"{y['bigram']} {y['t']:+.1f} ({y['share'][0]:.3f}/{y['share'][1]:.3f})"
                                           for y in x["kind_bigrams"][:6]))
        print("   tokens per stay:", ", ".join(f"{k} {a:.1f}/{b:.1f}" for k, (a, b) in x["kinds"].items()
                                              if k not in ("special", "prefix")))
        g = x["group_events"]
        print("   events per group: twins", {k: g["twins"][k] for k in ("mean", "p50", "p90", "single")},
              "real", {k: g["real"][k] for k in ("mean", "p50", "p90", "single")})

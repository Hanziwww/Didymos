"""The toolkit's command line (``configs/toolkit.yaml``; ``docs/toolkit.md``): one tool per call, its result as JSON
in its job directory and, with ``card=true``, the HTML card beside it; the result's path is printed.
"""

from __future__ import annotations

import json
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


def _hours(cfg):
    h = cfg.hours
    if h is None:
        raise SystemExit("hours= is required")
    return [float(x) for x in h] if OmegaConf.is_list(h) else float(h)


def _patient(s, cfg):
    if not cfg.patient:
        raise SystemExit("patient= is required (split:<dataset>/<split>/<stay id>, bank:<twin id>, meds:...)")
    return s.patient(str(cfg.patient), unknown_site=cfg.unknown_site)


def _save(s, res, cfg, name: str = "result"):
    from icutwin.toolkit import card

    results = res if isinstance(res, list) else [res]
    job = Path((results[0].get("settings") or {}).get("job") or s.paths.out / results[0]["tool"])
    job.mkdir(parents=True, exist_ok=True)
    out = job / f"{name}.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8", newline="\n")
    if cfg.card and results[0]["tool"] in card.SECTIONS:
        for i, r in enumerate(results):
            card.write(card.render(r, s if r.get("patient") else None), job / f"{name}_{i}.html")
    print(out)


def forecast(s, cfg):
    _save(s, s.forecast(_patient(s, cfg), _hours(cfg), mode=cfg.mode, n=cfg.n, cap_hours=float(cfg.cap_hours),
                        cap_mult=float(cfg.cap_mult), design=cfg.design, observed=bool(cfg.observed)), cfg)


def whatif(s, cfg):
    _save(s, s.whatif(_patient(s, cfg), _hours(cfg), str(cfg.treatment), n=int(cfg.n or 16)), cfg)


def explain(s, cfg):
    _save(s, s.explain(_patient(s, cfg), _hours(cfg), str(cfg.program), design=cfg.design,
                       generator_top=int(cfg.generator_top)), cfg)


def similar(s, cfg):
    _save(s, s.similar(_patient(s, cfg), _hours(cfg), index=cfg.index, k=int(cfg.k)), cfg)


def _spec(cfg) -> dict:
    import yaml

    if not cfg.spec:
        raise SystemExit("spec= (a YAML file) is required")
    return yaml.safe_load(Path(str(cfg.spec)).read_text(encoding="utf-8"))


def cohort(s, cfg):
    _save(s, s.cohort(_spec(cfg)), cfg)


def trial(s, cfg):
    _save(s, s.trial(_spec(cfg)), cfg)


def teach(s, cfg):
    sp = _spec(cfg)
    print(json.dumps(s.teach(sp.pop("objective"), **sp)["outputs"], indent=1))


def card_of(s, cfg):
    from icutwin.toolkit import card

    p = Path(str(cfg.result))
    res = json.loads(p.read_text(encoding="utf-8"))
    for i, r in enumerate(res if isinstance(res, list) else [res]):
        print(card.write(card.render(r, s if r.get("patient") else None), p.with_name(f"{p.stem}_{i}.html")))


def index(s, cfg):
    from icutwin.toolkit import similar as sm

    print(sm.build_index(s, str(cfg.name), str(cfg.anchors_dir), str(cfg.labels)))


TOOLS = {"forecast": forecast, "whatif": whatif, "explain": explain, "similar": similar, "cohort": cohort,
         "trial": trial, "teach": teach, "card": card_of, "index": index}


@hydra.main(version_base="1.3", config_path="../configs", config_name="toolkit")
def main(cfg: DictConfig) -> None:
    from icutwin.toolkit.session import Session

    if cfg.tool not in TOOLS:
        raise SystemExit(f"tool= one of {sorted(TOOLS)}")
    TOOLS[cfg.tool](Session.from_config(cfg), cfg)


if __name__ == "__main__":
    main()

#!/usr/bin/env bash
# icutwin on the AutoDL host: round 2 diagnosis, the header-order artefact. The audit's twins (copied before their
# deletion) with their headers in the spec's order (engine.HeaderOrder, lossless), audited and scored like the
# baseline; the copies deleted after. Logs: logs/canon_<step>.
BASE=/root/autodl-tmp
R=$BASE/icutwin
L=$BASE/logs
T3=$R/data/tokens/full_r3
E=$R/runs/eval/M32_main/auth
cd "$BASE" || exit 1
log() { echo "$(date '+%F %T') $*"; }
bash tools/run.sh canon_sort .venv/bin/python -c "
import pyarrow as pa, pyarrow.parquet as pq
from pathlib import Path
from icutwin.generate.engine import HeaderOrder
from icutwin.tokenization.vocab import Vocab
v = Vocab.load(Path('$T3/vocab.json'))
order = HeaderOrder(v)
src, dst = Path('$E/twins_canon_src'), Path('$E/twins_canon')
dst.mkdir(exist_ok=True)
for f in sorted(src.glob('*.parquet')):
    seqs = [order(x) for x in pq.read_table(f).column('ids').to_pylist()]
    pq.write_table(pa.table({'ids': pa.array(seqs, type=pa.list_(pa.int32()))}), dst / f.name)
    print(f.name, len(seqs), flush=True)
"
log "canon_sort exit $(cat "$L/canon_sort.exit")"
bash tools/run.sh canon_audit .venv/bin/python -m icutwin.eval.authenticity audit --tokens "$T3" \
  --twins "$E/twins_canon" --out "$E/audit_canon.json" &
bash tools/run.sh canon_quality .venv/bin/python -m icutwin.eval.quality evaluate --tokens "$T3" \
  --samples "$E/twins_canon" --out "$E/quality_canon.json" &
wait
log "canon_audit exit $(cat "$L/canon_audit.exit"); canon_quality exit $(cat "$L/canon_quality.exit")"
bash tools/run.sh canon_targets .venv/bin/python -m icutwin.eval.authenticity targets --audit "$E/audit_canon.json" \
  --quality "$E/quality_canon.json" --out "$E/targets_canon.json"
rm -rf "$E/twins_canon" "$E/twins_canon_src"
log "copies deleted"
echo 0 > "$L/canon.exit"

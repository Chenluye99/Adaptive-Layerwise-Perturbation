cd /home/zhang430/code/mismatch_rl && source ~/miniconda3/etc/profile.d/conda.sh && conda activate verl_new && python - <<'PY'
from pathlib import Path
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from run_scripts.prepare_data_guru import process_row, ARROW_SCHEMA

files = [
    Path('/home/zhang430/data/guru_rl92k/online_eval/online_eval/math__math_500.parquet'),
    Path('/home/zhang430/data/guru_rl92k/online_eval/online_eval/math__aime_repeated_8x_240.parquet'),
]
records = []
idx = 0
for split_file in files:
    df = pd.read_parquet(split_file)
    for row in df.to_dict(orient='records'):
        processed = process_row(row, idx=idx, split='online_eval')
        records.append(processed)
        idx += 1

schema = ARROW_SCHEMA
output_path = Path('/home/zhang430/data/guru_rl92k_prepared/math_eval.parquet')
output_path.parent.mkdir(parents=True, exist_ok=True)

if records:
    table = pa.Table.from_pylist(records, schema=schema)
    pq.write_table(table, output_path)
    print(f'Wrote {len(records)} rows to {output_path}')
else:
    print('No records written')
PY
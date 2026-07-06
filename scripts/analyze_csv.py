import csv
from collections import defaultdict
from pathlib import Path

data = []
csv_path = Path(__file__).resolve().parent.parent / 'results' / 'resultados.csv'
with open(csv_path) as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        if row[0] == 'Version':
            continue
        data.append(row)

h = {name: i for i, name in enumerate(header)}

agg = defaultdict(list)
for r in data:
    key = (r[h['Version']], r[h['Instancia']], r[h['KernelSize']], r[h['Scale']])
    agg[key].append({
        't_total': float(r[h['T_Total_ms']]),
        't_kernel': sum(float(r[h[f'T_{st}_kernel_ms']]) for st in ['Gray','Blur','Sobel','Resize']),
        't_h2d': sum(float(r[h[f'T_{st}_HtoD_ms']]) for st in ['Gray','Blur','Sobel','Resize']),
        't_d2h': sum(float(r[h[f'T_{st}_DtoH_ms']]) for st in ['Gray','Blur','Sobel','Resize']),
        'throughput': float(r[h['Throughput_MPps']]),
    })

result = {}
for k, vs in agg.items():
    n = len(vs)
    result[k] = {
        't_total': sum(v['t_total'] for v in vs) / n,
        't_kernel': sum(v['t_kernel'] for v in vs) / n,
        't_h2d': sum(v['t_h2d'] for v in vs) / n,
        't_d2h': sum(v['t_d2h'] for v in vs) / n,
        'throughput': sum(v['throughput'] for v in vs) / n,
    }

print()
print('=== TABLA PRINCIPAL ===')
print('Ver   Inst          K  S     T_total  T_kernel  T_h2d   T_d2h   Tput_MPps')
for k in sorted(result.keys()):
    v = result[k]
    print(f'{k[0]:<5}{k[1]:<14}{k[2]:<3}{k[3]:<6}{v["t_total"]:<8.3f}{v["t_kernel"]:<9.3f}{v["t_h2d"]:<7.3f}{v["t_d2h"]:<7.3f}{v["throughput"]:<8.1f}')

print()
print('=== SPEED-UP CUDA-Clasico vs CPU ===')
for inst in ['small','medium','large','no-divisible']:
    for k in ['5','9']:
        for s in ['0.5','1.75']:
            cpu = result.get(('CPU_Secuencial', inst, k, s))
            cuda = result.get(('CUDA_Clasico', inst, k, s))
            if cpu and cuda:
                print(f'  {inst:14} k={k} s={s}: speedup = {cpu["t_total"]/cuda["t_total"]:.1f}x')

print()
print('=== MEDIAS POR VERSION (todas las instancias juntas) ===')
for ver in ['CPU_Secuencial', 'CUDA_Clasico', 'CUDA_Tile', 'cuTile_Python']:
    items = [v for k, v in result.items() if k[0] == ver]
    if not items:
        continue
    n = len(items)
    t_total = sum(v['t_total'] for v in items) / n
    t_kernel = sum(v['t_kernel'] for v in items) / n
    t_h2d = sum(v['t_h2d'] for v in items) / n
    t_d2h = sum(v['t_d2h'] for v in items) / n
    tput = sum(v['throughput'] for v in items) / n
    print(f'  {ver:<18} T_total={t_total:7.2f}ms  T_kernel={t_kernel:7.2f}ms  T_h2d={t_h2d:7.2f}ms  T_d2h={t_d2h:7.2f}ms  Tput={tput:7.0f} MP/s')

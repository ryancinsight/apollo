import re
import os

path = 'd:/apollofft/crates/apollo-fft/src/application/execution/kernel/stockham/butterfly.rs'
code = open(path, 'r', encoding='utf-8').read()

# Custom parser for butterfly.rs since functions aren't just unsafe
blocks = []
current = []
for line in code.splitlines():
    if line.startswith('#[cfg(') or line.startswith('#[allow(') or line.startswith('#[inline]') or line.startswith('pub(super) unsafe fn') or line.startswith('pub(super) fn') or line.startswith('macro_rules!'):
        if current and len(current) > 5 and not any(l.startswith('use ') for l in current):
            blocks.append('\n'.join(current))
            current = []
    current.append(line)
if current:
    blocks.append('\n'.join(current))

block_map = {}
for i, b in enumerate(blocks):
    m = re.search(r'fn ([a-zA-Z0-9_]+)', b, re.M)
    if not m: m = re.search(r'macro_rules! ([a-zA-Z0-9_]+)', b, re.M)
    
    if m:
        block_map[m.group(1)] = b
    else:
        block_map[f'header_{i}'] = b

f64 = ['hybrid_radix8x512_64_avx_fma', 'fixed_len512_avx_fma']
f32 = ['hybrid_radix8x512_32_avx_fma', 'fixed_len512_32_avx_fma', 'fixed_len8_32_avx_fma', 'fixed_len4_32_avx_fma']
stage = ['stage_triple_scalar_one_impl', 'stage_triple_impl', 'stockham_quad_unrolled', 'stage_quad_impl', 'stage_pair_impl']
twiddles = ['build_butterfly512_twiddles_32', 'build_butterfly512_twiddles_64']
core = ['forward64_avx_with_scratch', 'forward32_avx_with_scratch']

def write_file(filepath, imports, fn_list):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(imports + '\n\n')
        for name in fn_list:
            if name in block_map:
                f.write(block_map[name] + '\n\n')
            else:
                print('Warning: not found:', name)
    # Check length
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = len(f.readlines())
        print(f'{filepath} -> {lines} lines')

base_dir = 'd:/apollofft/crates/apollo-fft/src/application/execution/kernel/stockham/butterfly'

header = '''use num_complex::{Complex32, Complex64};
use super::super::precision::{F64StockhamAvxFma, F32StockhamAvxFma};
use super::super::avx::*;
use super::super::transform::{transform, transform_len4096_four_triples};
'''

write_file(f'{base_dir}/f64.rs', header, f64)
write_file(f'{base_dir}/f32.rs', header, f32)
write_file(f'{base_dir}/stage.rs', header, stage)
write_file(f'{base_dir}/twiddles.rs', header, twiddles)
write_file(f'{base_dir}/core.rs', header, core)

mod = '''pub(crate) mod f64;
pub(crate) mod f32;
pub(crate) mod stage;
pub(crate) mod twiddles;
pub(crate) mod core;

pub(super) use core::*;
pub(super) use stage::*;
'''
with open(f'{base_dir}/mod.rs', 'w', encoding='utf-8') as f: f.write(mod)
print("done butterfly")

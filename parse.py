import re
import os

path = 'd:/apollofft/crates/apollo-fft/src/application/execution/kernel/stockham/avx_kernels.rs'
code = open(path, 'r', encoding='utf-8').read()

blocks = []
current = []
for line in code.splitlines():
    if line.startswith('#[cfg(target_arch = "x86_64")]'):
        if current:
            blocks.append('\n'.join(current))
            current = []
    current.append(line)
if current:
    blocks.append('\n'.join(current))

block_map = {}
for i, b in enumerate(blocks):
    m = re.search(r'^(?:pub\(super\)\s+)?unsafe fn ([a-zA-Z0-9_]+)', b, re.M)
    if m:
        block_map[m.group(1)] = b
    else:
        block_map[f'header_{i}'] = b

f64_base = ['stage64_avx_fma', 'stage64_groups_one_avx_fma']
f64_pair = ['stage_pair64_avx_fma', 'stage_pair64_groups_two_avx_fma', 'stage_pair64_radix1_avx_fma']
f64_triple_1 = ['stage_triple64_radix1_avx_fma', 'stage_triple64_quarter_groups_one_avx_fma', 'stage_triple64_low_live_avx_fma']
f64_triple_2 = ['stage_triple64_groups_eight_avx_fma', 'stage_triple64_throughput_avx_fma']
f64_quad = ['stockham_quad_store_pair64', 'stockham_quad_groups_eight64_first_pairs', 'stockham_quad_groups_eight64_second_pairs', 'stockham_quad_groups_eight64_low_live']
f64_fixed = ['cmul_vec64', 'avx_cmul_by_pair_twiddle', 'avx_rotate_quarter_turn', 'avx_butterfly2_pair', 'avx_butterfly4_pair', 'avx_butterfly8_pair', 'avx_transpose8_pairs', 'avx_twiddle_len64_pair_const', 'avx_apply_len64_twiddles_for_column_pair', 'fixed_len64_first_phase_column_pair', 'fixed_len64_second_phase_column_pair', 'fixed_len64_avx_fma']

f32_base = ['stage32_avx_fma', 'stage32_groups_one_avx_fma']
f32_pair = ['stage_pair32_avx_fma', 'stage_pair32_radix1_avx_fma', 'stage_pair32_quarter_groups_two_avx_fma', 'stage_pair32_groups_two_avx_fma']
f32_triple_1 = ['stage_triple32_radix1_avx_fma', 'stage_triple32_avx_fma', 'stage_triple32_low_live_avx_fma']
f32_triple_2 = ['stage_triple32_quarter_groups_two_avx_fma', 'stage_triple32_quarter_groups_one_avx_fma']
f32_quad = ['stockham_quad_split_pair32', 'stockham_quad_store_pair32', 'stockham_quad_groups_eight32']
f32_fixed = ['fixed_len64_32_avx_fma', 'cmul_vec32', 'cmul_pair32', 'store_complex32_low', 'store_complex32_high', 'avx_rotate_quarter_turn32']

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

base_dir = 'd:/apollofft/crates/apollo-fft/src/application/execution/kernel/stockham/avx'

header_64 = "use num_complex::Complex64;\nuse super::super::avx_kernels::cmul_vec64;"
header_32 = "use num_complex::Complex32;\nuse super::super::avx_kernels::cmul_vec32;"
# Wait, I am removing avx_kernels entirely. So imports need to reference the local modules.
header_64 = "use num_complex::Complex64;"
header_32 = "use num_complex::Complex32;"

write_file(f'{base_dir}/f64/base.rs', header_64 + "\nuse super::fixed::cmul_vec64;", f64_base)
write_file(f'{base_dir}/f64/pair.rs', header_64 + "\nuse super::fixed::cmul_vec64;", f64_pair)
write_file(f'{base_dir}/f64/triple_1.rs', header_64 + "\nuse super::fixed::{cmul_vec64, avx_rotate_quarter_turn};", f64_triple_1)
write_file(f'{base_dir}/f64/triple_2.rs', header_64 + "\nuse super::fixed::{cmul_vec64, avx_rotate_quarter_turn};\nuse super::quad::stockham_quad_store_pair64;", f64_triple_2)
write_file(f'{base_dir}/f64/quad.rs', header_64 + "\nuse super::fixed::{cmul_vec64, avx_rotate_quarter_turn};", f64_quad)
write_file(f'{base_dir}/f64/fixed.rs', header_64, f64_fixed)

write_file(f'{base_dir}/f32/base.rs', header_32 + "\nuse super::fixed::{cmul_vec32, cmul_pair32};", f32_base)
write_file(f'{base_dir}/f32/pair.rs', header_32 + "\nuse super::fixed::cmul_vec32;", f32_pair)
write_file(f'{base_dir}/f32/triple_1.rs', header_32 + "\nuse super::fixed::{cmul_vec32, avx_rotate_quarter_turn32};", f32_triple_1)
write_file(f'{base_dir}/f32/triple_2.rs', header_32 + "\nuse super::fixed::{cmul_vec32, avx_rotate_quarter_turn32};", f32_triple_2)
write_file(f'{base_dir}/f32/quad.rs', header_32 + "\nuse super::fixed::{cmul_vec32, avx_rotate_quarter_turn32, stockham_quad_split_pair32, stockham_quad_store_pair32};", f32_quad)
write_file(f'{base_dir}/f32/fixed.rs', header_32 + "\nuse super::super::super::stage::stage_triple32_radix1_avx_fma;\nuse super::triple_2::stage_triple32_quarter_groups_one_avx_fma;", f32_fixed)

# Now generate mod.rs
mod_f64 = '''pub(crate) mod base;
pub(crate) mod fixed;
pub(crate) mod pair;
pub(crate) mod quad;
pub(crate) mod triple_1;
pub(crate) mod triple_2;

pub(super) use base::*;
pub(super) use fixed::*;
pub(super) use pair::*;
pub(super) use quad::*;
pub(super) use triple_1::*;
pub(super) use triple_2::*;
'''
with open(f'{base_dir}/f64/mod.rs', 'w', encoding='utf-8') as f: f.write(mod_f64)

mod_f32 = '''pub(crate) mod base;
pub(crate) mod fixed;
pub(crate) mod pair;
pub(crate) mod quad;
pub(crate) mod triple_1;
pub(crate) mod triple_2;

pub(super) use base::*;
pub(super) use fixed::*;
pub(super) use pair::*;
pub(super) use quad::*;
pub(super) use triple_1::*;
pub(super) use triple_2::*;
'''
with open(f'{base_dir}/f32/mod.rs', 'w', encoding='utf-8') as f: f.write(mod_f32)

mod_avx = '''pub(crate) mod f32;
pub(crate) mod f64;

pub(super) use f32::*;
pub(super) use f64::*;
'''
with open(f'{base_dir}/mod.rs', 'w', encoding='utf-8') as f: f.write(mod_avx)

print("done writing to kernel/stockham/avx")

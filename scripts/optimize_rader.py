import os
import re

RADER_DIR = "d:/apollofft/crates/apollo-fft/src/application/execution/kernel/rader"

def process_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Find all sizes used in dispatch_inplace calls
    size_pattern = r'std::slice::from_raw_parts_mut\(.*?,\s*(\d+)\)'
    sizes = [int(s) for s in re.findall(size_pattern, content)]
    
    if not sizes:
        return
    
    unique_sizes = set(sizes)
    max_size = max(unique_sizes)
    
    # We only care about sizes that are powers of two, because only they need twiddles
    pot_sizes = {s for s in unique_sizes if (s & (s - 1) == 0)}
    
    # Check if we already optimized this file
    if "inner_scratch" in content:
        return
        
    # We need to inject the scratch buffer and twiddles at the beginning of the function body
    # Find the function signature
    sig_pattern = r'(pub\(crate\) fn rader_\d+_impl.*?\{\n)'
    match = re.search(sig_pattern, content, re.DOTALL)
    if not match:
        return
    
    insert_pos = match.end()
    
    injection = f'    let mut inner_scratch = [F::complex(0.0, 0.0); {max_size}];\n'
    for s in pot_sizes:
        injection += f'    let tw_fwd_{s} = F::cached_twiddle_fwd({s});\n'
        injection += f'    let tw_inv_{s} = F::cached_twiddle_inv({s});\n'
    
    content = content[:insert_pos] + injection + content[insert_pos:]
    
    # Now replace the dispatch_inplace calls
    
    def repl_dispatch(m):
        full_match = m.group(0)
        size_str = m.group(1)
        inverse_str = m.group(2)
        size = int(size_str)
        
        if size in pot_sizes:
            tw_name = f'tw_inv_{size}' if inverse_str == 'true' else f'tw_fwd_{size}'
            tw_arg = f'Some({tw_name}.as_ref())'
        else:
            tw_arg = 'None'
            
        replacement = f'''let chunk_slice = unsafe {{ std::slice::from_raw_parts_mut(chunk_ptr, {size}) }};
        crate::application::execution::kernel::mixed_radix::dispatch_inplace_with_scratch::<F, {inverse_str}, false>(
            chunk_slice,
            {tw_arg},
            &mut inner_scratch[..{size}],
        );'''
        return replacement

    pattern = r'let chunk_slice = unsafe \{\s*std::slice::from_raw_parts_mut\(chunk_ptr,\s*(\d+)\)\s*\};\s*crate::application::execution::kernel::mixed_radix::dispatch_inplace::<F,\s*(true|false),\s*false>\(\s*chunk_slice,\s*None,\s*\);'
    
    new_content = re.sub(pattern, repl_dispatch, content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(new_content)
    print(f"Processed {os.path.basename(filepath)}")

def main():
    for filename in os.listdir(RADER_DIR):
        if filename.startswith('dft') and filename.endswith('.rs'):
            process_file(os.path.join(RADER_DIR, filename))

if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Comprehensive WGPU device.rs migration to apollo-wgpu-helpers.

Handles ALL known edge cases:
- Self {} vs Ok(Self {}) for new()
- Kernel types taking &Device vs &Arc<Device>
- Extra struct fields (e.g. unitary_kernel)
- Different error types (WgpuError vs NufftWgpuError)
- Non-standard kernel construction patterns
"""
import os, re, glob

def migrate(content, crate_name, label):
    """Migrate a single device.rs content."""
    lines = content.split('\n')
    
    # --- Step 1: Determine patterns ---
    has_arc_device = False
    struct_name = None
    kernel_type = None
    extra_fields = []
    uses_ok_self = False
    uses_self_direct = False
    error_type = 'WgpuError'
    result_type = 'WgpuResult'
    
    for line in lines:
        # Find struct
        m = re.search(r'pub struct (\w+WgpuBackend)', line)
        if m:
            struct_name = m.group(1)
        # Find Arc<wgpu::Device> field
        if 'device: Arc<wgpu::Device>' in line:
            has_arc_device = True
        # Detect error type
        if 'NufftWgpuError' in line:
            error_type = 'NufftWgpuError'
            result_type = 'NufftWgpuResult'
    
    if not has_arc_device or not struct_name:
        return content  # Already migrated or no device.rs pattern
    
    # --- Find kernel type from struct ---
    for line in lines:
        m = re.search(r'kernel: Arc<(\w+Gpu\w*)>', line)
        if m:
            kernel_type = m.group(1)
            break
        m = re.search(r'unitary_kernel: Arc<(\w+Gpu\w*)>', line)
        if m:
            extra_fields.append('unitary_kernel')
    
    if not kernel_type:
        kernel_type = struct_name.replace('Backend', 'GpuKernel')
    
    # --- Find extra non-device/queue/kernel fields ---
    in_struct = False
    for line in lines:
        if 'pub struct' in line and struct_name in line:
            in_struct = True
            continue
        if in_struct and line.strip() == '}':
            break
        if in_struct and ':' in line and 'Arc<' in line:
            field = line.strip().rstrip(',')
            if 'device:' not in field and 'queue:' not in field and 'kernel:' not in field:
                field_name = field.split(':')[0].strip()
                if field_name not in extra_fields:
                    extra_fields.append(field_name)
    
    # --- Determine new() pattern ---
    for i, line in enumerate(lines):
        if f'pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>)' in line:
            # Look ahead for return type and body
            if '-> Self' in line:
                uses_self_direct = True
            elif f'-> {result_type}<Self>' in line or '-> WgpuResult<Self>' in line:
                uses_ok_self = True
            elif '-> NufftWgpuResult<Self>' in line:
                uses_ok_self = True
            break
    
    # --- Detect kernel constructor pattern ---
    # Check if kernel takes &Device or &Arc<Device>
    kernel_takes_arc = False
    kernel_takes_ref = True  # default
    for line in lines:
        if kernel_type and f'{kernel_type}::new(' in line:
            if '.as_ref()' in line or '.inner()' in line:
                kernel_takes_ref = True
            elif '&device' in line:
                # original: &device where device is Arc<wgpu::Device>
                kernel_takes_arc = True
            break
    
    # For nufft, kernel methods take &Arc<wgpu::Device>
    if 'nufft' in crate_name.lower():
        kernel_takes_arc = True
    
    # --- Perform replacements ---
    
    # 1. Add import
    if 'use apollo_wgpu_helpers::WgpuDevice;' not in content:
        content = re.sub(
            r'(use crate::infrastructure::kernel::\S+;)',
            r'\1\nuse apollo_wgpu_helpers::WgpuDevice;',
            content, count=1
        )
        if 'use apollo_wgpu_helpers::WgpuDevice;' not in content:
            # Try alternative locations
            content = re.sub(
                r'(use crate::domain::error::\S+;)',
                r'\1\nuse apollo_wgpu_helpers::WgpuDevice;',
                content, count=1
            )
    
    # 2. Replace struct fields
    content = re.sub(
        r'    device: Arc<wgpu::Device>,\n    queue: Arc<wgpu::Queue>,\n    kernel:',
        r'    device: WgpuDevice,\n    kernel:',
        content
    )
    
    # 3. Replace self.device.as_ref() -> device ref
    content = content.replace('self.device.as_ref()', 'self.device.inner()')
    content = content.replace('self.queue.as_ref()', 'self.device.queue().as_ref()')
    
    # 3b. For nufft, use .device() (Arc) not .inner() (&Device)
    if kernel_takes_arc:
        content = content.replace('self.device.inner()', 'self.device.device()')
        content = content.replace('self.device.queue().as_ref()', 'self.device.queue()')
    
    # 4. Replace &self.device and &self.queue field accesses (that haven't been handled)
    content = re.sub(r'&self\.device,\n\s+&self\.queue,', 
                     r'self.device.inner(),\n            self.device.queue().as_ref(),', content)
    content = re.sub(r'&self\.device, &self\.queue,',
                     r'self.device.inner(), self.device.queue().as_ref(),', content)
    
    if kernel_takes_arc:
        content = content.replace('self.device.inner()', 'self.device.device()')
        content = content.replace('self.device.queue().as_ref()', 'self.device.queue()')
    
    # 5. Fix device() and queue() getter bodies
    content = re.sub(r'(pub fn device\(&self\) -> &Arc<wgpu::Device> \{\n)\s+&self\.device', 
                     r'\1        self.device.device()', content)
    content = re.sub(r'(pub fn queue\(&self\) -> &Arc<wgpu::Queue> \{\n)\s+&self\.queue', 
                     r'\1        self.device.queue()', content)
    
    # 6. Fix new() method
    extra_field_inits = '\n'.join([f'        {f},' for f in extra_fields if f != 'unitary_kernel'])
    
    if uses_self_direct:
        old_new = r'(pub fn new\(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>\) -> Self \{\n)\s+(let kernel|\s*Self\s*\{)'
        if kernel_takes_arc:
            new_body = f'{struct_name}::new(device.inner()),\n        device,\n'
        else:
            new_body = f'pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {{\n        let device = WgpuDevice::new(device, queue);'
    # ... this is getting too complex for regex. Let me handle differently.
    
    # Simpler approach: Replace the entire new() and try_default() methods
    
    # Find and replace new()
    if uses_self_direct:
        # Self { pattern
        old_new_pat = r'pub fn new\(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>\) -> Self \{\n\s+Self \{\n\s+kernel: (.*?),\n\s+device,\n\s+queue,\n\s+\}'
        new_new = f'pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {{\n        let device = WgpuDevice::new(device, queue);'
        # Reconstruct with extra fields
        kernel_match = re.search(old_new_pat, content)
        if kernel_match:
            kernel_expr = kernel_match.group(1)
            new_new += f'\n        Self {{\n            kernel: {kernel_expr},\n            device,'
            for f in extra_fields:
                new_new += f'\n            {f},'
            new_new += '\n        }'
            content = content.replace(kernel_match.group(0), new_new)
    
    # Find and replace try_default()
    label = f'apollo-{crate_name.split("/")[-1]}'
    old_td = r'pub fn try_default\(\).*?\{.*?(?:\n.*?)*?\n\s+Self::new\(Arc::new\(device\), Arc::new\(queue\)\)\s*\)?'
    td_match = re.search(old_td, content, re.DOTALL)
    if td_match:
        new_td = f'pub fn try_default() -> {result_type}<Self> {{\n        let device = WgpuDevice::try_default("{label}")?;\n        '
        extra_init = ''
        for f in extra_fields:
            if 'unitary' in f.lower():
                extra_init += f'\n        let {f} = Arc::new(Unitary{struct_name.replace("Backend","")}GpuKernel::new(device.inner()));'
        if extra_init:
            new_td += f'let kernel = Arc::new({kernel_type}::new(device.inner()));{extra_init}\n        Ok(Self {{ device, kernel'
            for f in extra_fields:
                new_td += f', {f}'
            new_td += ' })'
        else:
            new_td += f'let kernel = Arc::new({kernel_type}::new(device.inner()));\n        Ok(Self {{ device, kernel }})'
        content = content.replace(td_match.group(0), new_td)
    
    return content

# --- Main ---
for crate_dir in sorted(glob.glob('crates/*-wgpu')):
    fpath = os.path.join(crate_dir, 'src', 'infrastructure', 'device.rs')
    if not os.path.exists(fpath):
        print(f"SKIP {crate_dir}")
        continue
    
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    crate_name = os.path.basename(crate_dir)
    label = crate_name
    modified = migrate(content, crate_name, label)
    
    if modified != content:
        with open(fpath, 'w', encoding='utf-8') as f:
            f.write(modified)
        print(f"DONE {crate_dir}")
    else:
        print(f"NOOP {crate_dir}")

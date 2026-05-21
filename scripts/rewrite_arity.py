import sys

filepath = 'd:/apollofft/crates/apollo-fft/src/application/execution/kernel/radix_composite/arity.rs'
with open(filepath, 'r', encoding='utf-8') as f:
    content = f.read()

# Replace trait definitions
content = content.replace(
'''    fn compute_group<F: CompositeCache + ShortWinogradScalar>(
        src: &[Complex<F>],
        dst: &mut [Complex<F>],
        prev_len: usize,
        b_out: usize,
        groups_out: usize,
        twiddles: &[&[Complex<F>]],
        tw_idx: usize,
        inverse: bool,
    );''',
'''    fn compute_group<F: CompositeCache + ShortWinogradScalar>(
        src: &[Complex<F>],
        dst: &mut [Complex<F>],
        prev_len: usize,
        b_out: usize,
        groups_out: usize,
        twiddles: &[&[Complex<F>]],
        tw_idx: usize,
        pointwise: Option<&[Complex<F>]>,
        inverse: bool,
    );'''
)

content = content.replace(
'''    fn compute_group_with_scratch<F: CompositeCache + ShortWinogradScalar>(
        src: &[Complex<F>],
        dst: &mut [Complex<F>],
        scratch: &mut [Complex<F>],
        prev_len: usize,
        b_out: usize,
        groups_out: usize,
        twiddles: &[&[Complex<F>]],
        tw_idx: usize,
        inverse: bool,
    );''',
'''    fn compute_group_with_scratch<F: CompositeCache + ShortWinogradScalar>(
        src: &[Complex<F>],
        dst: &mut [Complex<F>],
        scratch: &mut [Complex<F>],
        prev_len: usize,
        b_out: usize,
        groups_out: usize,
        twiddles: &[&[Complex<F>]],
        tw_idx: usize,
        pointwise: Option<&[Complex<F>]>,
        inverse: bool,
    );'''
)

# Replace Radix impl compute_group_with_scratch (calls compute_group)
content = content.replace(
'''    fn compute_group_with_scratch<F: CompositeCache + ShortWinogradScalar>(
        src: &[Complex<F>],
        dst: &mut [Complex<F>],
        _scratch: &mut [Complex<F>],
        prev_len: usize,
        b_out: usize,
        groups_out: usize,
        twiddles: &[&[Complex<F>]],
        tw_idx: usize,
        inverse: bool,
    ) {
        Self::compute_group(src, dst, prev_len, b_out, groups_out, twiddles, tw_idx, inverse);
    }''',
'''    fn compute_group_with_scratch<F: CompositeCache + ShortWinogradScalar>(
        src: &[Complex<F>],
        dst: &mut [Complex<F>],
        _scratch: &mut [Complex<F>],
        prev_len: usize,
        b_out: usize,
        groups_out: usize,
        twiddles: &[&[Complex<F>]],
        tw_idx: usize,
        pointwise: Option<&[Complex<F>]>,
        inverse: bool,
    ) {
        Self::compute_group(src, dst, prev_len, b_out, groups_out, twiddles, tw_idx, pointwise, inverse);
    }'''
)

# Now we need to update Compose impl
content = content.replace(
'''    fn compute_group<F: CompositeCache + ShortWinogradScalar>(
        src: &[Complex<F>],
        dst: &mut [Complex<F>],
        prev_len: usize,
        b_out: usize,
        groups_out: usize,
        twiddles: &[&[Complex<F>]],
        tw_idx: usize,
        inverse: bool,
    ) {''',
'''    fn compute_group<F: CompositeCache + ShortWinogradScalar>(
        src: &[Complex<F>],
        dst: &mut [Complex<F>],
        prev_len: usize,
        b_out: usize,
        groups_out: usize,
        twiddles: &[&[Complex<F>]],
        tw_idx: usize,
        pointwise: Option<&[Complex<F>]>,
        inverse: bool,
    ) {'''
)

content = content.replace(
'''    fn compute_group_with_scratch<F: CompositeCache + ShortWinogradScalar>(
        src: &[Complex<F>],
        dst: &mut [Complex<F>],
        scratch: &mut [Complex<F>],
        prev_len: usize,
        b_out: usize,
        groups_out: usize,
        twiddles: &[&[Complex<F>]],
        tw_idx: usize,
        inverse: bool,
    ) {''',
'''    fn compute_group_with_scratch<F: CompositeCache + ShortWinogradScalar>(
        src: &[Complex<F>],
        dst: &mut [Complex<F>],
        scratch: &mut [Complex<F>],
        prev_len: usize,
        b_out: usize,
        groups_out: usize,
        twiddles: &[&[Complex<F>]],
        tw_idx: usize,
        pointwise: Option<&[Complex<F>]>,
        inverse: bool,
    ) {'''
)

# In Compose: calling Inner
content = content.replace(
'''            Inner::compute_group(
                src,
                scratch,
                prev_len,
                b_out,
                groups_out * Outer::R_TOTAL,
                twiddles,
                tw_idx,
                inverse,
            );''',
'''            Inner::compute_group(
                src,
                scratch,
                prev_len,
                b_out,
                groups_out * Outer::R_TOTAL,
                twiddles,
                tw_idx,
                pointwise,
                inverse,
            );'''
)
content = content.replace(
'''            Inner::compute_group_with_scratch(
                src,
                scratch,
                inner_scratch,
                prev_len,
                b_out,
                groups_out * Outer::R_TOTAL,
                twiddles,
                tw_idx,
                inverse,
            );''',
'''            Inner::compute_group_with_scratch(
                src,
                scratch,
                inner_scratch,
                prev_len,
                b_out,
                groups_out * Outer::R_TOTAL,
                twiddles,
                tw_idx,
                pointwise,
                inverse,
            );'''
)

# In Compose: calling Outer
content = content.replace(
'''        Outer::compute_group_with_scratch(
            scratch,
            dst,
            inner_scratch,
            prev_len * Inner::R_TOTAL,
            b_out,
            groups_out,
            twiddles,
            tw_idx + Inner::DEPTH,
            inverse,
        );''',
'''        Outer::compute_group_with_scratch(
            scratch,
            dst,
            inner_scratch,
            prev_len * Inner::R_TOTAL,
            b_out,
            groups_out,
            twiddles,
            tw_idx + Inner::DEPTH,
            None,
            inverse,
        );'''
)
content = content.replace(
'''        Outer::compute_group(
            scratch,
            dst,
            prev_len * Inner::R_TOTAL,
            b_out,
            groups_out,
            twiddles,
            tw_idx + Inner::DEPTH,
            inverse,
        );''',
'''        Outer::compute_group(
            scratch,
            dst,
            prev_len * Inner::R_TOTAL,
            b_out,
            groups_out,
            twiddles,
            tw_idx + Inner::DEPTH,
            None,
            inverse,
        );'''
)

# And in composite_fused_adaptive:
content = content.replace(
'''pub(super) fn composite_fused_adaptive<F: CompositeCache + ShortWinogradScalar>(
    src: &[Complex<F>],
    dst: &mut [Complex<F>],
    prev_len: usize,
    b_out: usize,
    groups_out: usize,
    radices: &[usize],
    twiddles: &[&[Complex<F>]],
    inverse: bool,
) {''',
'''pub(super) fn composite_fused_adaptive<F: CompositeCache + ShortWinogradScalar>(
    src: &[Complex<F>],
    dst: &mut [Complex<F>],
    prev_len: usize,
    b_out: usize,
    groups_out: usize,
    radices: &[usize],
    twiddles: &[&[Complex<F>]],
    pointwise: Option<&[Complex<F>]>,
    inverse: bool,
) {'''
)

# And in macro
content = content.replace(
'''            if $depth == radices.len() {
                return $stage::compute_group::<F>(
                    src, dst, prev_len, b_out, groups_out, twiddles, 0, inverse,
                );
            }''',
'''            if $depth == radices.len() {
                return $stage::compute_group::<F>(
                    src, dst, prev_len, b_out, groups_out, twiddles, 0, pointwise, inverse,
                );
            }'''
)
content = content.replace(
'''                return $stage::compute_group_with_scratch::<F>(
                    src, dst, &mut scratch, prev_len, b_out, groups_out, twiddles, 0, inverse,
                );''',
'''                return $stage::compute_group_with_scratch::<F>(
                    src, dst, &mut scratch, prev_len, b_out, groups_out, twiddles, 0, pointwise, inverse,
                );'''
)

content = content.replace(
'''                buf0[k] = *unsafe { src.get_unchecked(idx) };''',
'''                let mut v = *unsafe { src.get_unchecked(idx) };
                if let Some(pw) = pointwise {
                    v = v * unsafe { *pw.get_unchecked(idx) };
                }
                buf0[k] = v;'''
)

content = content.replace(
'''                buf1[k] = *unsafe { src.get_unchecked(idx) };''',
'''                let mut v = *unsafe { src.get_unchecked(idx) };
                if let Some(pw) = pointwise {
                    v = v * unsafe { *pw.get_unchecked(idx) };
                }
                buf1[k] = v;'''
)

content = content.replace(
'''                buf2[k] = *unsafe { src.get_unchecked(idx) };''',
'''                let mut v = *unsafe { src.get_unchecked(idx) };
                if let Some(pw) = pointwise {
                    v = v * unsafe { *pw.get_unchecked(idx) };
                }
                buf2[k] = v;'''
)

content = content.replace(
'''                buf3[k] = *unsafe { src.get_unchecked(idx) };''',
'''                let mut v = *unsafe { src.get_unchecked(idx) };
                if let Some(pw) = pointwise {
                    v = v * unsafe { *pw.get_unchecked(idx) };
                }
                buf3[k] = v;'''
)

content = content.replace(
'''            buf[k] = *unsafe { src.get_unchecked(k * stride + src_base + j) };''',
'''            let idx = k * stride + src_base + j;
            let mut v = *unsafe { src.get_unchecked(idx) };
            if let Some(pw) = pointwise {
                v = v * unsafe { *pw.get_unchecked(idx) };
            }
            buf[k] = v;'''
)

with open(filepath, 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated signatures in arity.rs")

// ── twiddle-array builders ────────────────────────────────────────
macro_rules! tw2 {
    () => {{
        let o2 = stage_offsets[stage_idx + 1];
        let p2 = prev_len * r1;
        let r2 = radices[stage_idx + 1];
        [tw1, &all_twiddles[o2..o2 + (r2 - 1) * p2]]
    }};
}
macro_rules! tw3 {
    () => {{
        let o2 = stage_offsets[stage_idx + 1];
        let p2 = prev_len * r1;
        let r2 = radices[stage_idx + 1];
        let o3 = stage_offsets[stage_idx + 2];
        let p3 = p2 * r2;
        let r3 = radices[stage_idx + 2];
        [tw1, &all_twiddles[o2..o2 + (r2 - 1) * p2], &all_twiddles[o3..o3 + (r3 - 1) * p3]]
    }};
}
macro_rules! tw4 {
    () => {{
        let o2 = stage_offsets[stage_idx + 1];
        let p2 = prev_len * r1;
        let r2 = radices[stage_idx + 1];
        let o3 = stage_offsets[stage_idx + 2];
        let p3 = p2 * r2;
        let r3 = radices[stage_idx + 2];
        let o4 = stage_offsets[stage_idx + 3];
        let p4 = p3 * r3;
        let r4 = radices[stage_idx + 3];
        [
            tw1,
            &all_twiddles[o2..o2 + (r2 - 1) * p2],
            &all_twiddles[o3..o3 + (r3 - 1) * p3],
            &all_twiddles[o4..o4 + (r4 - 1) * p4],
        ]
    }};
}
macro_rules! tw5 {
    () => {{
        let o2 = stage_offsets[stage_idx + 1];
        let p2 = prev_len * r1;
        let r2 = radices[stage_idx + 1];
        let o3 = stage_offsets[stage_idx + 2];
        let p3 = p2 * r2;
        let r3 = radices[stage_idx + 2];
        let o4 = stage_offsets[stage_idx + 3];
        let p4 = p3 * r3;
        let r4 = radices[stage_idx + 3];
        let o5 = stage_offsets[stage_idx + 4];
        let p5 = p4 * r4;
        let r5 = radices[stage_idx + 4];
        [
            tw1,
            &all_twiddles[o2..o2 + (r2 - 1) * p2],
            &all_twiddles[o3..o3 + (r3 - 1) * p3],
            &all_twiddles[o4..o4 + (r4 - 1) * p4],
            &all_twiddles[o5..o5 + (r5 - 1) * p5],
        ]
    }};
}
macro_rules! tw6 {
    () => {{
        let o2 = stage_offsets[stage_idx + 1];
        let p2 = prev_len * r1;
        let r2 = radices[stage_idx + 1];
        let o3 = stage_offsets[stage_idx + 2];
        let p3 = p2 * r2;
        let r3 = radices[stage_idx + 2];
        let o4 = stage_offsets[stage_idx + 3];
        let p4 = p3 * r3;
        let r4 = radices[stage_idx + 3];
        let o5 = stage_offsets[stage_idx + 4];
        let p5 = p4 * r4;
        let r5 = radices[stage_idx + 4];
        let o6 = stage_offsets[stage_idx + 5];
        let p6 = p5 * r5;
        let r6 = radices[stage_idx + 5];
        [
            tw1,
            &all_twiddles[o2..o2 + (r2 - 1) * p2],
            &all_twiddles[o3..o3 + (r3 - 1) * p3],
            &all_twiddles[o4..o4 + (r4 - 1) * p4],
            &all_twiddles[o5..o5 + (r5 - 1) * p5],
            &all_twiddles[o6..o6 + (r6 - 1) * p6],
        ]
    }};
}

// ── fuse helpers ─────────────────────────────────────────────────
macro_rules! fuse2 {
    ($p:literal, $q:literal) => {
        if r1 == $p
            && stage_idx + 1 < radices.len()
            && radices[stage_idx + 1] == $q
            && prev_len * ($p * $q) <= FUSE_THRESHOLD
        {
            let tw = tw2!();
            if src_is_data {
                stockham_stage_fused::<F, Compose<Radix<$p>, Radix<$q>>>(
                    data, scratch, prev_len, &tw, inverse,
                );
            } else {
                stockham_stage_fused::<F, Compose<Radix<$p>, Radix<$q>>>(
                    scratch, data, prev_len, &tw, inverse,
                );
            }
            src_is_data = !src_is_data;
            prev_len *= $p * $q;
            stage_idx += 2;
            continue;
        }
    };
}
macro_rules! fuse3 {
    ($p:literal, $q:literal, $r:literal) => {
        if r1 == $p
            && stage_idx + 2 < radices.len()
            && radices[stage_idx + 1] == $q
            && radices[stage_idx + 2] == $r
            && prev_len * ($p * $q * $r) <= FUSE_THRESHOLD
        {
            let tw = tw3!();
            if src_is_data {
                stockham_stage_fused::<
                    F,
                    Compose<Compose<Radix<$p>, Radix<$q>>, Radix<$r>>,
                >(data, scratch, prev_len, &tw, inverse);
            } else {
                stockham_stage_fused::<
                    F,
                    Compose<Compose<Radix<$p>, Radix<$q>>, Radix<$r>>,
                >(scratch, data, prev_len, &tw, inverse);
            }
            src_is_data = !src_is_data;
            prev_len *= $p * $q * $r;
            stage_idx += 3;
            continue;
        }
    };
}
macro_rules! fuse4 {
    ($p:literal, $q:literal, $r:literal, $s:literal) => {
        if r1 == $p
            && stage_idx + 3 < radices.len()
            && radices[stage_idx + 1] == $q
            && radices[stage_idx + 2] == $r
            && radices[stage_idx + 3] == $s
            && prev_len * ($p * $q * $r * $s) <= FUSE_THRESHOLD
        {
            let tw = tw4!();
            if src_is_data {
                stockham_stage_fused::<
                    F,
                    Compose<Compose<Compose<Radix<$p>, Radix<$q>>, Radix<$r>>, Radix<$s>>,
                >(data, scratch, prev_len, &tw, inverse);
            } else {
                stockham_stage_fused::<
                    F,
                    Compose<Compose<Compose<Radix<$p>, Radix<$q>>, Radix<$r>>, Radix<$s>>,
                >(scratch, data, prev_len, &tw, inverse);
            }
            src_is_data = !src_is_data;
            prev_len *= $p * $q * $r * $s;
            stage_idx += 4;
            continue;
        }
    };
}
macro_rules! fuse5 {
    ($p:literal, $q:literal, $r:literal, $s:literal, $t:literal) => {
        if r1 == $p
            && stage_idx + 4 < radices.len()
            && radices[stage_idx + 1] == $q
            && radices[stage_idx + 2] == $r
            && radices[stage_idx + 3] == $s
            && radices[stage_idx + 4] == $t
            && prev_len * ($p * $q * $r * $s * $t) <= FUSE_THRESHOLD
        {
            let tw = tw5!();
            type C5Node = Compose<Compose<Compose<Compose<Radix<$p>, Radix<$q>>, Radix<$r>>, Radix<$s>>, Radix<$t>>;
            if src_is_data {
                stockham_stage_fused::<F, C5Node>(data, scratch, prev_len, &tw, inverse);
            } else {
                stockham_stage_fused::<F, C5Node>(scratch, data, prev_len, &tw, inverse);
            }
            src_is_data = !src_is_data;
            prev_len *= $p * $q * $r * $s * $t;
            stage_idx += 5;
            continue;
        }
    };
}
macro_rules! fuse6 {
    ($p:literal, $q:literal, $r:literal, $s:literal, $t:literal, $u:literal) => {
        if r1 == $p
            && stage_idx + 5 < radices.len()
            && radices[stage_idx + 1] == $q
            && radices[stage_idx + 2] == $r
            && radices[stage_idx + 3] == $s
            && radices[stage_idx + 4] == $t
            && radices[stage_idx + 5] == $u
            && prev_len * ($p * $q * $r * $s * $t * $u) <= FUSE_THRESHOLD
        {
            let tw = tw6!();
            type C6Node = Compose<Compose<Compose<Compose<Compose<Radix<$p>, Radix<$q>>, Radix<$r>>, Radix<$s>>, Radix<$t>>, Radix<$u>>;
            if src_is_data {
                stockham_stage_fused::<F, C6Node>(data, scratch, prev_len, &tw, inverse);
            } else {
                stockham_stage_fused::<F, C6Node>(scratch, data, prev_len, &tw, inverse);
            }
            src_is_data = !src_is_data;
            prev_len *= $p * $q * $r * $s * $t * $u;
            stage_idx += 6;
            continue;
        }
    };
}

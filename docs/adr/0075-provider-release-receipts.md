# ADR 0075: Coordinated provider release receipts

- Status: Accepted
- Date: 2026-10-15
- Items: APOLLO-PROVIDER-RELEASE-PREREQUISITES [patch]; APOLLO-PYTHON-NATIVE-LINK [patch]

## Decision and source identity

Publish the existing 19-package pending set after the compatible Eunomia 0.9.2
and correctly versioned Hephaestus 0.21.0 prerequisites. Preserve the merged
Moirai migration, transform algorithms and separate owner's Mellin regression.
The binding 0.2.1 remains unpublished on PyPI.

[Reviewed head](https://github.com/ryancinsight/apollo/commit/b33d00325892b205939af3933cb9e7488db62c02)
and [published merge](https://github.com/ryancinsight/apollo/commit/97236f7db26d579aecdb5aaf1134ef0fb0dc88ed)
share an [identical source tree](https://github.com/ryancinsight/apollo/tree/5622e19abe5b2cf6386393e01ef76712c8e1cba7).
All 14 actual hosted checks and the complete unchanged tracked source gate pass.
The blocking [release validation](https://github.com/ryancinsight/apollo/actions/runs/38680459498)
passes the direct full API comparison and FFT package dry-run. Baselines build;
no unavailable-baseline fallback supplies the verdict. Standard packaging
verifies all 19 pending libraries before the guarded merge.

All 1634 enabled native cases pass with the same 15 pre-existing skips; strict
all-feature/all-target Clippy, warning-denied docs and doctests pass. Both native
CI lanes execute the NumPy boundary regression. The corrected optimized ARM
ABI3 wheel passes all 35 smoke cases, has an aligned Mach-O string table and
no libpython dependency. Wheel SHA-256:
`66f410615ac58d616a10910e0f9bf54a6845a12b29bc2a8adf1b91bf625151dc`.

## Immutable registry receipts

All entries are independently verified present and not yanked on 2026-10-15.
Every trusted-publisher receipt identifies the published merge above and
[run 38681655413](https://github.com/ryancinsight/apollo/actions/runs/38681655413).
The existing publisher performs its planned-set package/API validation before
upload; no duplicate release, credential expansion or review override occurs.

| Crate | Version | Registry SHA-256 |
| --- | --- | --- |
| `apollo-czt` | 0.11.1 | `2b4a08fc2547e56ebf7e7d5d26527aa420eb47a84a9affac23b4022ffd7d9d1a` |
| `apollo-dctdst` | 0.11.1 | `1c66782445e90bca8dfe2fcbe4d47bf0265b65c1b217c80bc04ffbf8b87076e5` |
| `apollo-dht` | 0.11.1 | `a24e33cef72e02541fcf2d96fc6b7308329dbdb30bf2761555625dc0423542a9` |
| `apollo-fft` | 0.28.2 | `8ba91a8ce963961e584f392d2262b5d6d675c3d7f9b64bab90c06d17b6be86d2` |
| `apollo-frft` | 0.10.1 | `ba1bd78fbea283a4846d0acc16f266a24dd7ebd59913b60de4e47d7df266bfc2` |
| `apollo-fwht` | 0.10.1 | `2884e9c71670239bd95fe8c0d802ce9d6a8136b56f0954ffba0eb0cd9adc37b3` |
| `apollo-gft` | 0.11.1 | `1e85532bc1bee374d979b3e4922af21d497ffc44cfeb832927275c5012189371` |
| `apollo-hilbert` | 0.13.1 | `ed75ac42b81c91eb7a850d83f5221585b0b21d3acbccec6c186b232f20a57408` |
| `apollo-mellin` | 0.12.1 | `7a27269ba6bf9238a4f702607e2c19f0152ebde1823e7863ce1ccb86f8c231b9` |
| `apollo-ntt` | 0.10.1 | `691392be8da846ce362416442d5afec0ee42402f3b8e7d4883d7c27d4f73d3c6` |
| `apollo-nufft` | 0.8.1 | `01bf44ab8516567d947e5951370569263a57eb47ae451df62e0947d667c6c4a6` |
| `apollo-qft` | 0.10.1 | `53c92c92b5ce32c9915ca16207c88afafd593784a8aedac7553a01bb08875c36` |
| `apollo-radon` | 0.11.1 | `53f3014d6115a62fe9282b989b5e9a5f62000fbe2fa24139a2733abb29878bb4` |
| `apollo-sdft` | 0.9.1 | `cba98890a4848eecd47e8ab721749d5ca7bca03b594aaa083a65c1933604ffca` |
| `apollo-sft` | 0.10.1 | `7881aea1d721084ca66e21e442cb7b76590c0147c7be97f64161b24835721334` |
| `apollo-sht` | 0.8.1 | `4081bf6051e429a604639074dc32d39f2b3adee072e0c8eb7707d0b57c6258b8` |
| `apollo-stft` | 0.10.1 | `7e345295e50aab1aacb0a5fe8f5ad97e91124ce9f26a4d7f2b29dfc63e2dd660` |
| `apollo-validation` | 0.4.1 | `e79651692287f64d99a011f302f839cc5a8c53e86f67cc3454d025a17514217c` |
| `apollo-wavelet` | 0.11.1 | `42ae68191e99a85a5184a464b55c8dbd29caf866d288db80dbd4bddde4ec0f0a` |

## Limits

Miri remains deferred and explicitly unverified. The separate held task is not
resumed, and no performance experiments occur. Native pass counts and intentional
skips do not imply untested hardware coverage. This documentation does not change
crate versions or request another upload or deployment.

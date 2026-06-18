# Models

Large `.keras` model files are managed with Git LFS. A GitHub Release zip can also be used as fallback.

## Included Model

```text
unet_baseline_2000_padding_512.keras
```

Original local source:

```text
processed_padding_512/models/unet_baseline_2000_padding_512.keras
```

Approximate size: 90.17 MB.

SHA256:

```text
5A9D6E9ABA1D0A8ADA7A220C4A4B7C0DCFB36FF3C6361CE680E63ABE543DF6F0
```

## Release Asset

Prepared local release asset:

```text
release_assets/lpei-escoliose-models-v1.zip
```

This zip is not committed to Git. It should be uploaded manually to a GitHub Release if release-based model download is desired.

SHA256 of the prepared zip:

```text
88FA25CC772CFCE0B28D0D272704A5C2C9DB65CA2BF8506FBCE88FEF758A4693
```

## Optional Model Not Included

```text
unet_train_full_padding_512.keras
```

Original local source:

```text
processed_padding_512/models/train_full/unet_train_full_padding_512.keras
```

Approximate size: 89.14 MB.

This model belongs to a recent experimental branch. It is not included in the first evaluation-pack version because it should not be presented as a direct replacement for the refined checkpoint205 result.

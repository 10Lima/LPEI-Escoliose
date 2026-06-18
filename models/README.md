# Modelos

Os modelos `.keras` devem ser geridos por Git LFS ou por GitHub Release.

## Checkpoint minimo recomendado

```text
unet_baseline_2000_padding_512.keras
```

Origem local:

```text
processed_padding_512/models/unet_baseline_2000_padding_512.keras
```

Tamanho aproximado: 90.17 MB.

Estado no staging:

- ficheiro copiado para `models/`;
- deve ser seguido por Git LFS no repositorio final;
- tambem deve ser incluido no zip de release `lpei-escoliose-models-v1.zip`.

SHA256:

```text
5A9D6E9ABA1D0A8ADA7A220C4A4B7C0DCFB36FF3C6361CE680E63ABE543DF6F0
```

## Release asset preparado

```text
release_assets/lpei-escoliose-models-v1.zip
```

SHA256 do zip:

```text
88FA25CC772CFCE0B28D0D272704A5C2C9DB65CA2BF8506FBCE88FEF758A4693
```

## Modelo opcional

```text
unet_train_full_padding_512.keras
```

Origem local:

```text
processed_padding_512/models/train_full/unet_train_full_padding_512.keras
```

Tamanho aproximado: 89.14 MB.

Este modelo deve ser apresentado como experiencia recente, nao como substituto direto do checkpoint205 refinado.

Estado: nao incluido nesta primeira versao do evaluation pack.

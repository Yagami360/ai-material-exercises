# BioNeMo Framework を使用して xxx

## 方法

1. GPU インスタンスを構築する

1. NVIDIA NGC にログインし、API キーを作成する

    https://ngc.nvidia.com/signin

    ログイン後に、User > Setup > Generate API Key から API キーを作成

    > NGC 上から docker pull するにに必要

1. BioNeMo Framework 用の docker image を pull する

    ```bash
    docker pull nvcr.io/nvidia/clara/bionemo-framework:2.7
    ```

1. BioNeMo Framework の docker コンテナを起動する

    ```bash
    docker run --rm -it --gpus all \
        nvcr.io/nvidia/clara/bionemo-framework:2.7 \
        /bin/bash
    ```

## 参考サイト

- https://docs.nvidia.com/bionemo-framework/latest/main/getting-started/access-startup/
- https://github.com/NVIDIA/bionemo-framework
- https://note.com/wandb_jp/n/nb54160daf871

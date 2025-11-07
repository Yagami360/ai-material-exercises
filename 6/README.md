# BioNeMo Framework を使用して AI 創薬モデル（ESM-2）の推論を行う

## 方法

1. GPU インスタンスを構築する

    A100 で動作することを確認

    > V100 だと以下のエラーが発生する
    > ```bash
    > [rank0]: RuntimeError: CUDA error: no kernel image is available for execution on the device
    > ```

1. NVIDIA NGC にログインし、API キーを作成する

    https://ngc.nvidia.com/signin

    ログイン後に、User > Setup > Generate API Key から API キーを作成

    > NGC 上から docker pull するにに必要

1. NGC に接続する

    ```bash
    docker login nvcr.io
    #Username: $oauthtoken
    #Password <insert NGC API token here>
    ```

1. BioNeMo Framework 用の docker image を pull する

    ```bash
    docker pull nvcr.io/nvidia/clara/bionemo-framework:2.7
    ```

<!--
1. 各種環境変数を設定する

    ```bash
    # BioNeMo Framework で共通の環境変数
    export WANDB_API_KEY='dummy'    # wandb の API キー（https://wandb.ai から取得）
    ```

    `.env` ファイルで定義するのでも OK
-->

1. BioNeMo Framework の docker コンテナへの接続を行う

    ```bash
    mkdir -p datasets
    mkdir -p results

    docker run --rm -it --gpus all \
        --network host \
        --shm-size=4g \
        -v ${PWD}/datasets:/workspace/bionemo2/datasets \
        -v ${PWD}/results:/workspace/bionemo2/results \
        -v ${PWD}/src:/workspace/bionemo2/src \
        nvcr.io/nvidia/clara/bionemo-framework:2.7 \
        /bin/bash
    ```

    https://github.com/NVIDIA/bionemo-framework のレポジトリのコードが存在するコンテナになっている

    ```bash
    root@011d2a38b429:/workspace/bionemo2# ls -al
    total 40
    drwxrwxrwx 1 root root 4096 Sep 30 02:17 .
    drwxrwxrwx 1 root root 4096 Sep 26 10:03 ..
    drwxrwxrwx 1 root root 4096 Sep 30 02:17 .cache
    drwxrwxrwx 1 root root 4096 Sep 26 19:23 LICENSE
    -rwxrwxrwx 1 root root 6358 Sep 30 02:13 README.md
    -rwxrwxrwx 1 root root    7 Sep 30 02:13 VERSION
    drwxrwxrwx 1 root root 4096 Sep 30 02:17 ci
    drwxrwxrwx 1 root root 4096 Sep 30 02:13 docs
    drwxrwxrwx 1 root root 4096 Sep 30 02:13 sub-packages
    ```
    ```bash
    root@011d2a38b429:/workspace/bionemo2# cd ..
    root@011d2a38b429:/workspace# ls -al
    total 28
    drwxrwxrwx 1 root root 4096 Sep 26 10:03 .
    drwxr-xr-x 1 root root 4096 Nov  6 08:31 ..
    -rw-rw-rw- 1 root root 2048 Jun 12 15:29 README.md
    drwxrwxrwx 1 root root 4096 Sep 30 02:17 bionemo2
    drwxrwxrwx 1 root root 4096 Jun 12 15:29 docker-examples
    -rw-rw-rw- 1 root root  467 Jun 12 06:42 license.txt
    drwxrwxrwx 1 root root 4096 Jun 12 15:33 tutorials
    ```

1. AI 創薬モデル（ESM-2）の学習済みモデルをダウンロードする

    コンテナ内にて、以下のコマンドを実行する

    ```bash
    cd src
    python download_checkpoint.py
    ```

    デフォルトでは、`/root/.cache/bionemo` ディレクトリ以下に学習済みモデルが保存される

1. AI 創薬モデル（ESM-2）の推論時の入力データを作成する

    コンテナ内にて、以下のコマンドを実行する

    ```bash
    cd src
    python create_dataset.py
    ```

    以下のようなタンパク質シーケンスデータを入力データを作成する

    ```csv
    sequences
    TLILGWSDKLGSLLNQLAIANESLGGGTIAVMAERDKEDMELDIGKMEFDFKGTSVI
    LYSGDHSTQGARFLRDLAENTGRAEYELLSLF
    GRFNVWLGGNESKIRQVLKAVKEIGVSPTLFAVYEKN
    DELTALGGLLHDIGKPVQRAGLYSGDHSTQGARFLRDLAENTGRAEYELLSLF
    KLGSLLNQLAIANESLGGGTIAVMAERDKEDMELDIGKMEFDFKGTSVI
    LFGAIGNAISAIHGQSAVEELVDAFVGGARISSAFPYSGDTYYLPKP
    LGGLLHDIGKPVQRAGLYSGDHSTQGARFLRDLAENTGRAEYELLSLF
    LYSGDHSTQGARFLRDLAENTGRAEYELLSLF
    ISAIHGQSAVEELVDAFVGGARISSAFPYSGDTYYLPKP
    SGSKASSDSQDANQCCTSCEDNAPATSYCVECSEPLCETCVEAHQRVKYTKDHTVRSTGPAKT
    ```

1. AI 創薬モデル（ESM-2）の推論を行う

    コンテナ内にて、以下のコマンドを実行する

    ```bash
    infer_esm2 \
        --checkpoint-path /root/.cache/bionemo/0798767e843e3d54315aef91934d28ae7d8e93c2849d5fcfbdf5fac242013997-esm2_650M_nemo2.tar.gz.untar \
        --data-path ./datasets/sequences.csv \
        --results-path ./results \
        --micro-batch-size 3 \
        --num-gpus 1 \
        --precision "bf16-mixed" \
        --include-hiddens \
        --include-embeddings \
        --include-logits \
        --include-input-ids
    ```
    ```bash
    ...
    LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]
    [NeMo W 2025-11-07 07:50:52 nemo_logging:405] Could not copy Trainer's 'max_steps' to LR scheduler's 'max_steps'. If you are not using an LR scheduler, this warning can safely be ignored.
    [NeMo I 2025-11-07 07:50:52 nemo_logging:393]  > number of parameters on (tensor, pipeline) model parallel rank (0 ,0): 651164288
    [NeMo I 2025-11-07 07:51:01 nemo_logging:393] Inference predictions are stored in results/predictions__rank_0__dp_rank_0.pt
        dict_keys(['token_logits', 'binary_logits', 'hidden_states', 'embeddings'])
    ```

    推論結果は、`--results-path` で指定したディレクトリ以下に保存される。

    ```bash
    root@sakai-gpu-dev:/workspace/bionemo2# cd results/
    root@sakai-gpu-dev:/workspace/bionemo2/results# ls -al
    total 28196
    drwxr-xr-x 2 1005 1006     4096 Nov  7 07:51 .
    drwxrwxrwx 1 root root     4096 Nov  7 07:50 ..
    -rw-r--r-- 1 root root 28863920 Nov  7 07:51 predictions__rank_0__dp_rank_0.pt
    ```

    `predictions__rank_0__dp_rank_0.pt` 内に、`dict_keys(['token_logits', 'binary_logits', 'hidden_states', 'embeddings'])` で各データが存在する模様

    > `infer_esm2 --help` コマンドで各種引数の意味を確認可能

    > `infer_esm2` コマンドの実装は、https://github.com/NVIDIA/bionemo-framework/blob/main/sub-packages/bionemo-esm2/src/bionemo/esm2/scripts/infer_esm2.py に存在する

1. 推論時の出力データを確認する

    ```bash
    cd src
    python check_results.py
    ```

    ```bash
    token_logits    torch.Size([1024, 10, 128])
    hidden_states   torch.Size([10, 1024, 1280])
    embeddings      torch.Size([10, 1280])
    ```

    形状の説明:

    - Embeddings shape: (10, 1280):

        - 10 は入力シーケンスの数（10行）。

        - 1280 は、ESM-2 650Mモデルの埋め込みベクトルの次元数。これは、シーケンス全体から平均化された単一のベクトル表現。

    - Hiddens shape: (10, 62, 1280):

        - 10 は入力シーケンスの数。

        - 62 は、最長のシーケンス長（60）に加えて、先頭のCLSトークンと末尾のEOSトークンの合計である62トークン。

        - 1280 は、各トークンの隠れ状態の次元数。

    - Logits shape: (10, 62, 33):

        - 10 と 62 は隠れ状態と同じ意味です。

        - 33 は、ESM-2モデルの語彙サイズです（20の標準アミノ酸、パディング、マスク、不明、特殊トークンを含む）。

## 参考サイト

- https://docs.nvidia.com/bionemo-framework/latest/main/examples/bionemo-esm2/inference/

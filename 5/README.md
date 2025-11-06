# BioNeMo Framework を使用して AI 創薬モデル（ESM-2）の学習を行う

## 方法

1. GPU インスタンスを構築する

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

1. 各種環境変数を設定する

    ```bash
    # BioNeMo Framework で共通の環境変数
    export NGC_CLI_API_KEY='dummy'  # NGC の API キー
    export WANDB_API_KEY='dummy'    # wandb の API キー
    ...
    ```

    .env ファイルでも OK

1. BioNeMo Framework の docker コンテナへの接続を行う

    - 環境変数なし（動作テスト用）

        ```bash
        docker run --rm -it --gpus all \
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

    - 環境変数あり

        ```bash
        docker run \
        --rm -it \
        --gpus all \
        --network host \
        --shm-size=4g \
        -e WANDB_API_KEY \
        -e NGC_CLI_API_KEY \
        -e NGC_CLI_ORG \
        -e NGC_CLI_TEAM \
        -e NGC_CLI_FORMAT_TYPE \
        -v $LOCAL_DATA_PATH:$DOCKER_DATA_PATH \
        -v $LOCAL_MODELS_PATH:$DOCKER_MODELS_PATH \
        -v $LOCAL_RESULTS_PATH:$DOCKER_RESULTS_PATH \
        nvcr.io/nvidia/clara/bionemo-framework:2.7 \
        /bin/bash
        ```

1. AI 創薬モデル（ESM-2）の学習を行う

    コンテナ内にて、以下のコマンドを実行する

    ```bash
    # ESM-2 用の環境変数
    export MY_DATA_SOURCE="ngc"

    # The fastest transformer engine environment variables in testing were the following two
    TEST_DATA_DIR=$(download_bionemo_data esm2/testdata_esm2_pretrain:2.0 --source $MY_DATA_SOURCE); \
    ESM2_650M_CKPT=$(download_bionemo_data esm2/650m:2.0 --source $MY_DATA_SOURCE); \

    train_esm2 \
        --train-cluster-path ${TEST_DATA_DIR}/2024_03_sanity/train_clusters_sanity.parquet \
        --train-database-path ${TEST_DATA_DIR}/2024_03_sanity/train_sanity.db \
        --valid-cluster-path ${TEST_DATA_DIR}/2024_03_sanity/valid_clusters.parquet \
        --valid-database-path ${TEST_DATA_DIR}/2024_03_sanity/validation.db \
        --result-dir ./results \
        --experiment-name test_experiment \
        --num-gpus 1 \
        --num-nodes 1 \
        --val-check-interval 10 \
        --num-dataset-workers 1 \
        --num-steps 10 \
        --max-seq-length 1024 \
        --limit-val-batches 2 \
        --micro-batch-size 2 \
        --restore-from-checkpoint-path ${ESM2_650M_CKPT}
    ```

    > GPU メモリとして、xxx GB 程度必要

    > `train_esm2 --help` コマンドで各種引数の意味を確認可能

    > `train_esm2` コマンドの実装は、https://github.com/NVIDIA/bionemo-framework/blob/main/sub-packages/bionemo-esm2/src/bionemo/esm2/scripts/train_esm2.py に存在する

## 参考サイト

- https://docs.nvidia.com/bionemo-framework/latest/main/getting-started/training-models/
- https://github.com/NVIDIA/bionemo-framework

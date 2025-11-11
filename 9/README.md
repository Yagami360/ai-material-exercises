# MONAI Toolkit でのCT画像セグメンテーション用モデル（VISTA-3D）を使用して、CT画像における特定の内蔵領域をアノテーションする（ファインチューニング+推論）

## 方法

1. GPU インスタンスを構築する

    システム要件: https://docs.nvidia.com/clara/monai/getting_started/requirements.html

1. JupyterLab をブラウザ表示するために、インスタンスの `8888` ポートを解放する

1. MONAI Toolkit の Docker image を pull する

    ```bash
    docker pull nvcr.io/nvidia/clara/monai-toolkit:3.0
    ```

1. MONAI Toolkit の Docker コンテナを起動する

    - JupyterLab で起動する場合

        ```bash
        docker run --gpus all -it --rm \
        --ipc=host --net=host \
        nvcr.io/nvidia/clara/monai-toolkit:3.0
        ```
        デフォルトで `8888` ポートが使用される。変更したい場合は `-e JUPYTER_PORT=8900` を追加

    - [Option] bash で起動する場合

        ```bash
        docker run --gpus all -it --rm \
            --ipc=host --net=host \
            nvcr.io/nvidia/clara/monai-toolkit:3.0 /bin/bash
        ```

1. ブラウザ上で JupyterLab を開く

1. JupyterLab 上で `tutorials/monai/vista_3d/vista3d_spleen_finetune.ipynb` 等の内容に従って処理を行う

    CT画像セグメンテーション用モデル（VISTA-3D）のチュートリアルである `http://localhost:8888/lab/tree/tutorials/monai/vista_3d/vista3d_spleen_finetune.ipynb` に従って、処理を行う

    - VISTA-3D

        CT画像セグメンテーション用モデルで、120以上の主要臓器クラスをセグメンテーション可能

    - 学習用データセット

        <img width="790" height="392" alt="Image" src="https://github.com/user-attachments/assets/ec0f1f5d-9669-4215-87d2-70516b03cd05" />

        腹部断面のCT画像と正解ラベル（黄色の領域：脾臓として正しくラベル付けされた領域）

        - 239 x 239 の画像
        - 枚数: 150枚（かなり少ないが、ファインチューニングなので問題なし？）
        - https://msd-for-monai.s3-us-west-2.amazonaws.com/Task09_Spleen.tar からダウンロード
            - http://medicaldecathlon.com/ で公開されているデータセット

    - 学習

        - 5 epoch のみ学習

        - 使用 GPU メモリ: 8 ~ 9GB 程度（T4でも動かせた）

        - loss 値のグラフ

            <img width="996" height="547" alt="Image" src="https://github.com/user-attachments/assets/13f50975-df08-4570-badb-448a1e8469c2" />

            今回は、チュートリアルと同じ設定のまましたが、epoch 5 では、まだ十分に loss が収束してないのでもっと Epoch 数増やしたほうが良さそう

    - ファインチューニング後の推論結果

        <img width="644" height="509" alt="Image" src="https://github.com/user-attachments/assets/20965a04-d11a-438f-974b-ebe29a0934ab" />

        腹部断面のCT画像（テスト用データを使用）から脾臓部分をうまくアノテーション（セグメンテーション）できている

## 参考サイト

- https://docs.nvidia.com/clara/monai/getting_started/quickstart.html
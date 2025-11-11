# MONAI Toolkit を起動する

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

1. JupyterLab 上で `welcome.md` 等の内容に従って処理を行う

    <img width="1439" height="712" alt="Image" src="https://github.com/user-attachments/assets/f87a88b1-7cab-420a-99c9-eb88336da631" />


## 参考サイト

- https://docs.nvidia.com/clara/monai/getting_started/quickstart.html
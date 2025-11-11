# MONAI Toolkit を使用して xxx

## 方法

1. MONAI Toolkit の Docker image を pull する

    ```bash
    ```

1. MONAI Toolkit の Docker コンテナを起動する

    - bash で起動する場合

        ```bash
        docker run  --gpus all -it --rm --ipc=host --net=host \
            nvcr.io/nvidia/clara/monai-toolkit \
            /bin/bash
        ```

    - JupyterLab で起動する場合

        ```bash
        docker run --gpus all -it --rm --ipc=host --net=host nvcr.io/nvidia/clara/monai-toolkit
        ```

## 参考サイト

- https://docs.nvidia.com/clara/monai/getting_started/quickstart.html
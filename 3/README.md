# NVIDIA BioNeMo の blueprint デモからAI創薬における入出力データを理解する

以下の NVIDIA BioNeMo の blueprint デモ（NVIDIA BioNeMo Blueprint）で、AI創薬における入出力データの意味を理解する

https://build.nvidia.com/nvidia/protein-binder-design-for-drug-discovery


## パイプラインの全体構成

```bash
1. RFdiffusion (創造): 「こういう機能に最適な構造をゼロから生成する。」
2. ProteinMPNN (翻訳): 「この生成された構造を安定させるためのアミノ酸配列を設計する。」
3. AlphaFold2 (検証): 「ProteinMPNNが設計した配列が、本当に意図した構造に折りたたまれるかを確認する。」
```

## 入力データ

- RFdiffusion

    <img width="300" alt="Image" src="https://github.com/user-attachments/assets/968cd6e2-8896-4f7a-9173-6a46a8458651" />

    <img width="300" alt="Image" src="https://github.com/user-attachments/assets/45554ab6-61da-480c-85b4-c59685587335" />

- ProteinMPNN

    <img width="300" alt="Image" src="https://github.com/user-attachments/assets/03a522f6-efca-4794-aa0e-bb64bc56a56f" />

    <img width="300" alt="Image" src="https://github.com/user-attachments/assets/13a2fd76-eaae-4a4f-82fd-f36ad3e7fba4" />

- AlphaFold2

    <img width="515" alt="Image" src="https://github.com/user-attachments/assets/05ac5fe9-c86c-4cb3-978a-49b6e0f6f44c" />

## 出力データ

- json データ

```json
{
    "result": {
        "alphafold2":
            {
                "all_predicted_pdbs": [
                    "ATOM      1  N   SER A  19      96.155  70.201  45.493  1.00100.66           N  \nATOM      2  CA  SER A  19      94.696  70.434  45.702  1.00101.02           C  \nATOM      3  C   SER A  19      93.987  69.087  45.880  1.00100.43           C  \nATOM      4  O   SER A  19      94.494  68.054  45.439  1.00 99.56           O  \nATOM      5  CB  SER A  19      94.116  71.194  44.499  1.00102.01           C  \nATOM      6  OG  SER A  19      92.778  71.609  44.730  1.00102.75           O  \nATOM      7  N   THR A  20      92.825  69.102  46.535  1.00100.27           N  \nATOM      8  CA  THR A  20      92.051  67.879  46.776  1.00 99.38           C  \nATOM      9  C   THR A  20      91.073  67.584  45.641  1.00 98.31           C  \nATOM     10  O   THR A  20      90.444  68.499  45.100  1.00 98.40           O  \nATOM     11  CB  THR A  20      91.236  67.973  48.092  1.00 98.99           C  \nATOM     12  OG1 THR A  20      92.126  68.161  49.199  1.00100.97           O  \nATOM     13  CG2 THR A  20      90.435  66.697  48.320  1.00 98.68           C  \nATOM     14  N   ILE A  21      90.946  66.305  45.289  1.00 97.23           N  \nATOM     15  CA  ILE A  21      90.028  65.874  44.232  1.00 96.91           C  \nATOM     16  C   ILE A  21      88.577  66.107  44.659  1.00 95.29           C  \nATOM     17  O   ILE A  21      87.687  65.320  44.350  1.00 95.10           O  \nATOM     18  CB  ILE A  21      90.209  64.370  43.890  1.00 97.49           C  \nATOM     19  CG1 ILE A  21      89.957  63.500  45.130  1.00 98.25  
                    ...
                },
                "proteinmpnn": {
                "mfasta": ">input, score=3.0430, global_score=2.2795, fixed_chains=['B'], designed_chains=['A'], CA_model_name=v_48_002, git_hash=unknown, seed=667\nGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGGG\n>T=0.1, sample=1, score=1.3999, global_score=1.9260, seq_recovery=0.0469\nGPEEEEKKREIEALVAEMAKQFEEVKPLIELLEELKKKIGEGEKEAEKELKKKIEEEEKRKEEA\n",
                "scores": [
                    1.3998868465423584
                ],
                "probabilities": [
                    [
                    [
                        2.9388796951579366e-10,
                        6.684248006694004e-19,
                        3.878053780881352e-11,
                        1.8682724638174886e-12,
                        2.496218801052632e-15,
                        0.9949813485145569,
                        8.619456505049819e-16,
                        2.2613524466467754e-15,
                        2.8787888739501e-10,
                        1.5028966569505253e-12,
                        0.004999660421162844,
                        5.45019411546388e-12,
                        1.129599669011383e-12,
                        4.379279345433822e-15,
                        8.465267549440103e-12,
                        0.000018977581930812448,
                        2.3997567821787413e-10,
                        3.865102394363202e-15,
                        1.6532286080523171e-18,
                        6.183128094975089e-17,
                        0
                    ],
                    [
                        0.00005282574420562014,
                        7.163673617923613e-18,
                        ...
                    ]
                    ],
        "rfdiffusion": {
            "output_pdb": "ATOM      1  N   GLY A   1     -41.231  -9.202  10.946  1.00  0.00\nATOM      2  CA  GLY A   1     -40.410  -8.027  11.210  1.00  0.00\nATOM      3  C   GLY A   1     -39.456  -8.271  12.373  1.00  0.00\nATOM      4  O   GLY A   1     -38.329  -7.777  12.377  1.00  0.00\nATOM      5  N   GLY A   2     -39.762  -9.223  13.182  1.00  0.00\nATOM      6  CA  GLY A   2     -38.935  -9.554  14.336  1.00  0.00\nATOM      7  C   GLY A   2     -37.653 -10.259  13.912  1.00  0.00\nATOM      8  O   GLY A   2     -36.582 -10.007  14.465  1.00  0.00\nATOM      9  N   GLY A   3     -37.722 -11.118  12.941  1.00  0.00\nATOM     10  CA  GLY A   3     -36.533 -11.809  12.458  1.00  0.00\nATOM     11  C   GLY A   3     -35.546 -10.833  11.830  1.00  0.00\nATOM     12  O   GLY A   3     -34.337 -10.939  12.036  1.00  0.00\nATOM     13  N   GLY A   4     -36.039  -9.888  11.050  1.00  0.00\nATOM     14  CA  GLY A   4   
            },
            ...
}
```

- 上記 json データを 3D レンダリングした結果
    
    <img width="300" alt="Image" src="https://github.com/user-attachments/assets/72540891-e03a-40d2-96dc-6513b7e9ee28" />

- RFdiffusion

    <img width="300" alt="Image" src="https://github.com/user-attachments/assets/9eacd37d-9a35-4707-8028-4561a8568c82" />

- ProteinMPNN

    <img width="300" alt="Image" src="https://github.com/user-attachments/assets/43fb9ec5-83a2-4869-ba49-04a5550e6660" />

- AlphaFold2

    <img width="300" alt="Image" src="https://github.com/user-attachments/assets/3c92ddeb-fd25-4993-9a4e-74cef1a80cf6" />

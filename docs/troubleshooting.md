# トラブルシューティング

## 問題の確認

[GitHub Issues](https://github.com/nvidia-cosmos/cosmos-predict2.5/issues) も確認してください。新規で Issue を作成する場合は、出力ディレクトリ全体をアップロードしてください。

### CUDA ドライバのバージョン不足

**対処:** [CUDA 12.8.1](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html#cuda-toolkit-major-component-versions) に対応した最新の NVIDIA ドライバへ更新してください。

ドライバの互換性を確認:

```shell
nvidia-smi | grep "CUDA Version:"
```

### メモリ不足（OOM）エラー

**対処:** 14B ではなく 2B モデルを使用する、マルチ GPU を利用する、あるいはバッチサイズ/解像度を下げてください。

## ガイド

### ログ

ログは `<output_dir>/*.log` に保存されます。

### プロファイリング

プロファイルを取得するには、`--profile` フラグを付与します。[pyinstrument](https://pyinstrument.readthedocs.io/en/latest/guide.html) のプロファイルが `<output_dir>/profile.pyisession` に出力されます。

プロファイルの表示:

```shell
pyinstrument --load=<output_dir>/profile.pyisession
```

プロファイルのエクスポート:

```shell
pyinstrument --load=<output_dir>/profile.pyisession -r html -o <output_dir>/profile.html
```

[pyinstrument](https://pyinstrument.readthedocs.io/en/latest/guide.html) も参照ください。

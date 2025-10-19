<p align="center">
    <img src="https://github.com/user-attachments/assets/28f2d612-bbd6-44a3-8795-833d05e9f05f" width="274" alt="NVIDIA Cosmos"/>
</p>

<p align="center">
  <a href="https://www.nvidia.com/en-us/ai/cosmos/">製品サイト</a>&nbsp | 🤗 <a href="https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B">Hugging Face</a>&nbsp | <a href="https://research.nvidia.com/publication/2025-09_world-simulation-video-foundation-models-physical-ai">論文</a> | <a href="https://research.nvidia.com/labs/dir/cosmos-transfer2.5/">論文サイト</a>
</p>

NVIDIA Cosmos™ は、物理 AI のために設計されたプラットフォームで、最先端の生成系ワールド基盤モデル（WFM）、堅牢なガードレール、加速化されたデータ処理・キュレーションパイプラインを備えています。実世界システムに特化して設計されており、自動運転（AV）、ロボティクス、ビデオ解析 AI エージェントなどの物理 AI アプリケーションの開発を加速します。

Cosmos のワールド基盤モデルは 3 種類のモデルタイプで提供され、いずれもポストトレーニングでカスタマイズ可能です: [cosmos-predict](https://github.com/nvidia-cosmos/cosmos-predict2.5), [cosmos-transfer](https://github.com/nvidia-cosmos/cosmos-transfer2.5), [cosmos-reason](https://github.com/nvidia-cosmos/cosmos-reason1)。

## お知らせ（News）

* [October 13, 2025] Transfer2.5 Auto Multiview の[ポストトレーニング用データセット](https://github.com/nvidia-cosmos/cosmos-transfer2.5/blob/main/docs/post-training_auto_multiview.md)を更新し、NVIDIA Blackwell をサポートするセットアップ依存関係を追加しました。
  
* [October 6, 2025] 次世代のワールドシミュレーションモデルである [Cosmos-Transfer2.5](https://github.com/nvidia-cosmos/cosmos-transfer2.5) と [Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5) をリリースしました！

* [June 12, 2025] Cosmos ファミリーの一部として [Cosmos-Transfer1-DiffusionRenderer](https://github.com/nv-tlabs/cosmos-transfer1-diffusion-renderer) を公開しました。

## Cosmos-Transfer2.5

Cosmos-Transfer2.5 は、RGB、Depth、Segmentation など複数のビデオモダリティの構造化入力を受け付けるマルチ ControlNet です。ユーザーは JSON ベースの controlnet_specs で生成を設定し、少ないコマンドで推論を実行できます。単一動画の推論、自動コントロールマップ生成、マルチ GPU 構成のいずれにも対応しています。

Physical AI は、2 つの重要なデータ拡張ワークフローで生成されたデータを用いて学習します。

### シミュレーションからフォトリアリズムへ

3D シミュレーションで高忠実度を達成する必要性を最小化します。

**入力プロンプト:**
> この動画はロボットマニピュレーションのデモンストレーションで、おそらく実験室やテスト環境で撮影されています。青い布を扱う 2 本のロボットアームが登場します。<details> <summary>クリックして詳細プロンプトを表示</summary>
> 背景にベージュのソファがある部屋で、ロボット作業の中立的な背景となっています。布は黄色いクッションの上に置かれ、ロボットアームは布の左右に配置されています。左のロボットアームは白で黒いグリッパ、右のアームは黒でより複雑な関節グリッパを備えています。序盤、布はクッションの上に広げられています。左のアームが布に近づき、位置決めのためにグリッパを開閉します。右のアームは当初静止しており、補助の準備をしています。進行につれ、左のアームが布をつかみ、クッションからわずかに持ち上げます。続いて右のアームが動き、反対側をつかめるようグリッパを調整します。2 本のアームは協調して布を持ち上げ保持します。布は精密に操作され、アームの巧緻性と制御が示されます。カメラは終始固定で、ロボットアームと布の相互作用に焦点を当て、作業における詳細な動きと協調を観察できます。</details>

| Input Video | Computed Control | Output Video |
| --- | --- | --- |
| <video src="https://github.com/user-attachments/assets/bffc031e-3933-4511-a659-136965931ab0" width="100%" alt="Input video" controls></video> | <video src="https://github.com/user-attachments/assets/8ed4c49c-af26-4318-b95a-32f9cf44d992" width="100%" alt="Control map video" controls></video> | <video src="https://github.com/user-attachments/assets/88f7e63b-efe1-46ff-8174-df2f01462c53" width="100%" alt="Output video" controls></video> |

### ワールド状態の多様性を拡張

センサーで取得した RGB や正解データの拡張を活用します。

**入力プロンプト:**
> この動画は現代的な都市環境でのドライブシーンで、車載カメラや固定カメラで撮影されたものと思われます。<details><summary>クリックして詳細プロンプトを表示</summary>
> ガラス張りの近代的な高層ビルが立ち並ぶ広い複数車線の道路が舞台です。前方には黒い車が一定の速度で走っており、その他の車はまばらです。カメラは固定され、車両の前進に伴い道路と周囲を一貫して映します。左側の歩道には木々が並び、都市景観に緑を添えています。歩道には歩行者も見られ、のんびり歩く人や建物の近くに立つ人がいます。建物は大きなガラス窓を持つものから、より伝統的なコンクリート外装のものまで様々です。商業的な看板やロゴも見られ、オフィスや店舗の存在を示しています。前方の道路には三角コーンが置かれており、道路工事や車線規制が行われていることを示唆し、車両に合流や車線変更を促しています。道路標示は明瞭で、白い矢印が進行方向を示しています。空は快晴で視界が良好です。動画全体を通して車両は一定速度を保ち、カメラは交差点へと近づく様子を捉えます。全体として、ピーク時以外の都市らしい落ち着いた雰囲気です。</details>

| Input Video | Computed Control | Output Video |
| --- | --- | --- |
| <video src="https://github.com/user-attachments/assets/4705c192-b8c6-4ba3-af7f-fd968c4a3eeb" width="100%" alt="Input video" controls></video> | <video src="https://github.com/user-attachments/assets/ba92fa5d-2972-463e-af2e-a637a810a463" width="100%" alt="Control map video" controls></video> | <video src="https://github.com/user-attachments/assets/0c5151d4-968b-42ad-a517-cdc0dde37ee5" width="100%" alt="Output video" controls></video> |

## Cosmos-Transfer2.5 モデルファミリー

Cosmos-Transfer は複数の業界分野でのデータ生成をサポートします。以下に概要を示します。今後も Transfer ファミリーに特化モデルを追加していく予定です！

[**Cosmos-Transfer2.5-2B**](docs/inference.md): 物理 AI とロボティクス向けに一から学習された汎用 [チェックポイント](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B)。

[**Cosmos-Transfer2.5-2B/auto**](docs/inference_auto_multiview.md): 自動運転用途に向けてポストトレーニングされた特化チェックポイント。[マルチビュー チェックポイント](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B/tree/main/auto)

## ユーザーガイド

* [セットアップガイド](docs/setup.md)
* [推論](docs/inference.md)
  * [Auto Multiview](docs/inference_auto_multiview.md)
* [ポストトレーニング](docs/post-training.md)
  * [Auto Multiview](docs/post-training_auto_multiview.md)

## コントリビューション

私たちはコミュニティの協力によって成長しています。[NVIDIA-Cosmos](https://github.com/nvidia-cosmos/) は皆さんの貢献なしには成り立ちません。[コントリビューションガイド](CONTRIBUTING.md) を確認して、ぜひ Issue などでフィードバックをお寄せください。

オープンソースの物理 AI の可能性を広げてくれている皆さんに、心から感謝します 🙏

## ライセンスと連絡先

本プロジェクトは、追加のサードパーティ製オープンソースソフトウェアをダウンロード・インストールします。使用前に各プロジェクトのライセンス条項をご確認ください。

NVIDIA Cosmos のソースコードは [Apache 2 License](https://www.apache.org/licenses/LICENSE-2.0) の下で公開されています。

NVIDIA Cosmos のモデルは [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license) の下で公開されています。カスタムライセンスについては [cosmos-license@nvidia.com](mailto:cosmos-license@nvidia.com) までお問い合わせください。

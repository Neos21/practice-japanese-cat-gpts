# Practice Japanese Cat GPTs

日本語対応のモデルをお試し。

それぞれの実行結果サンプルとモデルデータの容量はファイル内にコメントで記載。動作速度は Core i7-7700K・GTX1080・RAM 32GB 環境にて。

- 実行手順

```bash
$ poetry install
$ poetry run python ./practice/【ファイル名】.py
```

- 環境構築手順

```bash
$ python3 -V
Python 3.11.2

$ poetry -V
Poetry (version 1.8.3)

$ poetry new practice
$ poetry add torch transformers accelerate sentencepiece protobuf
```


## Links

- [Neo's World](https://neos21.net/)

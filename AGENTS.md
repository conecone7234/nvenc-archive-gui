# Repository Agent Instructions

このファイルは、このリポジトリで作業する Codex/AI エージェントが最初に確認する恒久メモです。
今後、ユーザー指示のうち継続して守るべきものはここへ追記します。

## 作業開始

- 作業開始前に毎回 `git pull` と `git fetch --prune` を実行する。
- 新規チャットでこのリポジトリを触る場合は、最新の `develop` から作業ブランチを自動作成する。
- ブランチ名は可能なら `codex/` プレフィックスを使う。ローカルの Git ref 制約で使えない場合は `codex-` プレフィックスにフォールバックする。

## コミットと公開

- このリポジトリでコードやドキュメントを生成・修正したら、検証後に自動で commit と push まで行う。
- protected branch へ直接 push できない場合は、専用ブランチへ push し、必要なら PR で進める。

## レビュー対策

- 毎回、完了前にレビューで指摘されそうな点を確認する。
- 指摘されそうな点が見当たらなくなったら、その旨をユーザーへ明示する。
- レビュー向けの説明、PRコメント、Copilotレビュー指示は日本語で書く。

## 検証

- Python 実行環境が不安定な場合は、まず利用可能な `python` / `py` / `sys.executable` を確認する。
  この Codex Desktop 環境では `C:\Users\conecone\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe` が使えることがあるが、これはローカル例として扱い、他環境で存在する前提にしない。
- 基本の検証順序は `compileall src tests`、`pytest -q`、`ruff check .`、`ruff format --check .`、`git diff --check`。
- UI関連の修正や指摘対応では、必ず一度GUIを実行して、スクリーンショットなどで見え方を確認する。

## アプリ固有の注意

- `resource_ids` と出力ごとの `backend` をエンコード先の正とする。`use_gpu` はレガシーのNVENC互換状態として扱う。
- NVENC、QSV、AMF、CPU を同一の「GPU」扱いに潰さず、それぞれの制約と表示を保つ。
- セグメント再開では、ジョブ開始時に決まった `segment_root` を使い、実行中にソース情報から出力先を再計算しない。
- ホーム画面は開始/一時停止/再開/停止、ログ、進捗に集中させ、詳細な出力設定はプロファイル編集に寄せる。

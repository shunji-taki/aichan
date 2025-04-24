# aichan
Slack bot that connect to OpenAI API

AIちゃん (aichan)

Slack 上で OpenAI API を活用する、会話型 AI ボットです。
会話履歴やシステムプロンプトの記録、統計管理、スレッドごとの文脈保持などを特徴としています。
Slack Bolt + OpenAI + SQLite3 によるシンプルかつ柔軟な構成です。

【構成ファイル】
	•	app.py : メインアプリケーション。Slackイベントやコマンド処理のロジックを含みます。
	•	requirements.txt : 必要な Python パッケージ一覧。

【主な機能】
	•	GPT-4.1（nano/mini含む）によるスレッド文脈を加味した対話
	•	システムプロンプトの保存・編集機能（スラッシュコマンドで設定可能）
	•	SQLite による会話履歴と統計の保存
	•	メンションやスレッド内での応答、時々発話機能（雑談型）
	•	トークン使用量と想定コストを表示（USD/JPY換算）
	•	モデル切替対応（gpt-4.1, gpt-4.1-mini, gpt-4.1-nano, dall-e-3）

【セットアップ方法】
	1.	リポジトリをクローン
git clone https://github.com/shunji-taki/aichan.git
cd aichan
	2.	Python 仮想環境を作成し、依存関係をインストール
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
	3.	.env ファイルを作成し、以下の環境変数を設定
SLACK_BOT_TOKEN=（Slack Bot Token）
SLACK_APP_TOKEN=（Socket Mode Token）
OPENAI_API_KEY=（OpenAI API Key）
BOSS_SLACK_USERID=（オーナーのSlack User ID）
	4.	アプリ起動
python app.py

【スラッシュコマンド一覧】
	•	/ai_replace_sysprompt : システムプロンプトを新しい内容に置き換える
	•	/ai_add_sysprompt     : システムプロンプトを追記する
	•	/ai-model             : OpenAIモデルを変更する
	•	/ai-tokens            : トークン使用状況とコスト統計を表示する

【データファイル（SQLite）】
	•	conversation.db : 会話ログの記録
	•	sysmemory.db    : システムプロンプトやトークン統計の保存

【依存ライブラリ（抜粋）】
	•	slack-bolt
	•	openai
	•	python-dotenv
	•	tiktoken

【今後の展望（例）】
	•	会話履歴のWeb可視化UI
	•	複数ユーザー対応 / プロファイルごとのプロンプト管理
	•	雑談トピック提案モジュールの実装

【ライセンス】

本プロジェクトのライセンスは未定です。再利用・商用利用については著作権者の許可を得てください。